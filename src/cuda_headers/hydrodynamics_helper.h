#ifndef HYDRODYNAMICS_HELPER_H
#define HYDRODYNAMICS_HELPER_H
#include "hydrodynamics.h"


//////////////////////////////////
////	   XSPH Variant		  ////
//////////////////////////////////
__global__ void __compute_x_factor(float3* x_factor, const SPH_variables* sph, const uint* cell_bounds,
	const particle_kinematics* kinematics, const particle* particles, const uint particle_capacity)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }
	if (!particles[idx].exists()) { x_factor[idx] = make_float3(0.f); return; }

	const float3 this_pos = particles[idx].true_pos();
	const float3 this_vel_factor = kinematics[idx].velocity_kms;
	const float this_radius_km = kinematics[idx].radius_km;
	const float this_dens = sph[idx].avg_density_kgm3;

	float3 average = make_float3(0.f);
	morton_cell_iterator iter = morton_cell_iterator(particles[idx].morton_index());

	FOREACH(uint, loop_morton, iter)
		for (uint i = __read_start_idx(cell_bounds, loop_morton), end = __read_end_idx(cell_bounds, loop_morton); i < end; i++)
		{
			float3 displacement = this_pos - particles[i].true_pos();
			const float sq_dst = dot(displacement, displacement);
			if (sq_dst >= __sq_dist_cutoff) { continue; }
			float radius_factor = ___radius_factor(kinematics[i].radius_km, this_radius_km);
			average += (___spline_kernel(sq_dst, radius_factor) * kinematics[i].mass_Tg * 2.f / (this_dens + sph[i].avg_density_kgm3)) * (kinematics[i].velocity_kms - this_vel_factor);
		}
	x_factor[idx] = average;
}
__global__ void __apply_x_factor(const float3* x_factor, particle_kinematics* kinematics, const uint particle_capacity, const float strength)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	kinematics[idx].velocity_kms += x_factor[idx] * strength;
}

void apply_xsph_variant(smart_gpu_buffer<float3>& temporary, hydrogravitational_simulation& simulation, const float timestep, float recenter_strength = 1.f, float strength = 1.f, bool apply_heat = true)
{
	dim3 threads(simulation.particle_capacity > 512u ? 512u : simulation.particle_capacity);
	dim3 blocks((uint)ceilf(simulation.particle_capacity / (float)threads.x));

	simulation.sort_spatially();
	simulation.generate_gravitational_data();
	simulation.apply_gravitation();
	simulation.compute_sph_quantities();

	__compute_x_factor<<<blocks, threads>>>(temporary.gpu_buffer_ptr, simulation.smoothed_particle_hydrodynamics.gpu_buffer_ptr, simulation.cell_bounds.gpu_buffer_ptr,
		simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particles.buffer.gpu_buffer_ptr, simulation.particle_capacity);
	__apply_x_factor<<<blocks, threads>>>(temporary.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particle_capacity, strength);

	simulation.apply_thermodynamic_timestep(timestep, apply_heat);

	if (recenter_strength > 0.f)
		simulation.apply_kinematics_recenter(timestep, recenter_strength);
	else
		simulation.apply_kinematics(timestep);

	__apply_x_factor<<<blocks, threads>>>(temporary.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particle_capacity, -strength);
}

//////////////////////////////////
////	   Timestepping		  ////
//////////////////////////////////

inline __device__ __host__ float __cfl_factor(const float3 rel_vel, const float3 rel_pos, const float radius_factor)
{
	const float displacement = dot(rel_pos, rel_pos);
	return (length(rel_vel) * sqrtf(displacement) - dot(rel_vel, rel_pos)) / (radius_factor + displacement);
}
__global__ void __courant_friedrich_lewy_condition_bulk(float* condition_buffer, const uint* cell_bounds, const particle_kinematics* kinematics, const particle* particles)
{
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= grid_cell_count) { return; }

	const uint start = __read_start_idx(cell_bounds, idx);
	const uint end = __read_end_idx(cell_bounds, idx);
	if (start == end) { condition_buffer[idx] = 0.f; return; }

	const float this_rad = kinematics[start].radius_km;
	const float3 this_pos = particles[start].true_pos();
	const float3 this_vel = kinematics[start].velocity_kms;

	float condition = 0.f;
	for (uint i = 0u; i < grid_dimension_pow - minimum_depth; i++) // look farther and farther out.
	{
		morton_cell_iterator iter = morton_cell_iterator(idx >> (3u * i), grid_dimension_pow - i);
		FOREACH(uint, loop_morton, iter) // quick and dirty check for *any* particle in a certain cell.
		{
			uint to_read = loop_morton << (3u * i);
			const uint start_t = __read_start_idx(cell_bounds, to_read);
			const uint end_t = __read_end_idx(cell_bounds, to_read);

			if (start_t == end_t) { continue; }
			to_read = (start_t + end_t) >> 1u;
			condition = fmaxf(condition, __cfl_factor(kinematics[to_read].velocity_kms - this_vel,
													  particles[to_read].true_pos() - this_pos,
									 ___radius_factor(kinematics[to_read].radius_km, this_rad)));
		}
	}
	condition_buffer[idx] = condition;
}
__global__ void __courant_friedrich_lewy_condition_hydrodynamics(float* condition_buffer, const uint* cell_bounds, const SPH_variables* averages, const particle_kinematics* kinematics)
{
	const uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= grid_cell_count) { return; }

	const uint index_from = (cell_bounds[grid_cell_count - 1u] * idx) / grid_cell_count;
	float factor = averages[index_from].speed_of_sound_kms / kinematics[index_from].radius_km;
	condition_buffer[idx] = factor;
}
struct timestep_helper
{
	smart_gpu_buffer<float> timestepping_buffer;
	timestep_helper() : timestepping_buffer(grid_cell_count) {

	}
	float maximal_timestep_courant_friedrich_lewy_condition_hydrodynamics(hydrodynamics_simulation& simulation)
	{
		dim3 threads = dim3(simulation.cell_bounds.dedicated_len > 512u ? 512u : simulation.cell_bounds.dedicated_len);
		dim3 blocks = dim3((uint)ceilf(simulation.cell_bounds.dedicated_len / (float)threads.x));
		__courant_friedrich_lewy_condition_hydrodynamics<<<blocks, threads>>>(timestepping_buffer.gpu_buffer_ptr, simulation.cell_bounds.gpu_buffer_ptr, simulation.smoothed_particle_hydrodynamics.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr);
		return 1.f / find_maximum_val(timestepping_buffer);
	}
	float maximal_timestep_courant_friedrich_lewy_condition_bulk(kinematic_simulation& simulation)
	{
		dim3 threads = dim3(simulation.cell_bounds.dedicated_len > 512u ? 512u : simulation.cell_bounds.dedicated_len);
		dim3 blocks = dim3((uint)ceilf(simulation.cell_bounds.dedicated_len / (float)threads.x));
		__courant_friedrich_lewy_condition_bulk<<<blocks, threads>>>(timestepping_buffer.gpu_buffer_ptr, simulation.cell_bounds.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particles.buffer.gpu_buffer_ptr);
		return .45f / find_maximum_val(timestepping_buffer);
	}
	float maximal_timestep_hydrodynamics_simulation(hydrodynamics_simulation& simulation)
	{
		return fminf(maximal_timestep_courant_friedrich_lewy_condition_hydrodynamics(simulation),
				maximal_timestep_courant_friedrich_lewy_condition_bulk(simulation));
	}
};

#endif