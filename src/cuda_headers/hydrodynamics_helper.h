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
			average += (___spline_kernel(sq_dst, radius_factor) * kinematics[i].mass_Tg * 2.f / (this_dens + sph[idx].avg_density_kgm3)) * (kinematics[i].velocity_kms - this_vel_factor);
		}
	x_factor[idx] = average;
}
__global__ void __apply_x_factor(const float3* x_factor, particle_kinematics* kinematics, const uint particle_capacity, const float strength)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	kinematics[idx].velocity_kms += x_factor[idx] * strength;
}
void apply_xsph_variant(smart_gpu_buffer<float3>& temporary, hydrogravitational_simulation& simulation, const float timestep, float recenter_strength = 1.f, float strength = 1.f)
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

	simulation.apply_thermodynamic_timestep(timestep);

	if (recenter_strength > 0.f)
		simulation.apply_kinematics_recenter(timestep, recenter_strength);
	else
		simulation.apply_kinematics(timestep);

	__apply_x_factor<<<blocks, threads>>>(temporary.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particle_capacity, -strength);
}
void apply_damping_stabilizer(smart_gpu_buffer<float3>& temporary, hydrogravitational_simulation& simulation, const float timestep, float recenter_strength = 1.f, float strength = 1.f)
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

	simulation.apply_thermodynamic_timestep(timestep, false);

	if (recenter_strength > 0.f)
		simulation.apply_kinematics_recenter(timestep, recenter_strength);
	else
		simulation.apply_kinematics(timestep);
}


#endif