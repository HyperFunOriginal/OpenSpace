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
	return (length(rel_vel) * sqrtf(displacement) - dot(rel_vel, rel_pos)) / (radius_factor * .5f + displacement);
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
	
	const uint index_from = (uint)((cell_bounds[grid_cell_count - 1u] * (ulong)idx) / grid_cell_count); // need to prevent overflow
	float factor = averages[index_from].speed_of_sound_kms /
		fmaxf(0.62035049089f * cbrtf(kinematics[index_from].mass_Tg / averages[index_from].avg_density_kgm3),
			kinematics[index_from].radius_km); // smoother and more reliable than pure radius
	condition_buffer[idx] = factor;
}
struct timestep_helper
{
	smart_gpu_buffer<float> timestepping_buffer;
	void destroy()
	{
		timestepping_buffer.destroy();
	}
	timestep_helper() : timestepping_buffer(grid_cell_count) {

	}
	float maximal_timestep_courant_friedrich_lewy_condition_hydrodynamics(hydrodynamics_simulation& simulation)
	{
		dim3 threads = dim3(simulation.cell_bounds.dedicated_len > 512u ? 512u : simulation.cell_bounds.dedicated_len);
		dim3 blocks = dim3((uint)ceilf(simulation.cell_bounds.dedicated_len / (float)threads.x));
		__courant_friedrich_lewy_condition_hydrodynamics<<<blocks, threads>>>(timestepping_buffer.gpu_buffer_ptr, simulation.cell_bounds.gpu_buffer_ptr, simulation.smoothed_particle_hydrodynamics.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr);
		return .7f / find_maximum_val(timestepping_buffer);
	}
	float maximal_timestep_courant_friedrich_lewy_condition_bulk(kinematic_simulation& simulation)
	{
		dim3 threads = dim3(simulation.cell_bounds.dedicated_len > 512u ? 512u : simulation.cell_bounds.dedicated_len);
		dim3 blocks = dim3((uint)ceilf(simulation.cell_bounds.dedicated_len / (float)threads.x));
		__courant_friedrich_lewy_condition_bulk<<<blocks, threads>>>(timestepping_buffer.gpu_buffer_ptr, simulation.cell_bounds.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particles.buffer.gpu_buffer_ptr);
		return .35f / find_maximum_val(timestepping_buffer);
	}
	float maximal_timestep_hydrodynamics_simulation(hydrodynamics_simulation& simulation)
	{
		float hydro = maximal_timestep_courant_friedrich_lewy_condition_hydrodynamics(simulation);
		float bulk = maximal_timestep_courant_friedrich_lewy_condition_bulk(simulation);
		return fminf(hydro, bulk);
	}
};

//////////////////////////////////
////  Hydrostatic Equilibria  ////
//////////////////////////////////

/// <summary>
/// Contains physical quantities arranged in strata based on physical distance from an origin.
/// </summary>
struct stratification_data
{
	smart_gpu_cpu_buffer<float> data;
	float outer_radius;

	stratification_data() : data(), outer_radius() {}
	stratification_data(uint layers, float outer_radius) : data(layers), outer_radius(outer_radius) {}
	void destroy()
	{
		data.destroy();
	}
};

/// <summary>
/// Constructs an object to determine the radius and density of an object in hydrostatic equilibrium given a mass and material. Assumes a uniform temperature throughout.
/// Uses Regula Falsi to determine the correct density profile, with preprocessing for bounds.
/// </summary>
struct hydrostatic_body_solver
{
	float radius_km;
	float mass_Tg;
	float temperature_K;
	float core_density_kgm3;
	material_properties& mat;
private:
	float total_mass_from_core_density(float core_density)
	{
		float ln_rho = logf(core_density);
		float mass_contained_within = mass_Tg * .00005f;
		radius_km = cbrtf(0.23873241463f * mass_contained_within / fmaxf(core_density, mat.standard_density_kgm3));
		const float max_dr = radius_km;

		while (ln_rho > 2.f && radius_km < domain_size_km)
		{
			const float rho = expf(ln_rho);
			float dPdrho = mat.EOS_pressure_GPa(rho / mat.standard_density_kgm3, rho / mat.molar_mass_kgmol, temperature_K) > 0.f ? 
				mat.EOS_dp_drho_isothermal(rho, rho / mat.standard_density_kgm3, rho / mat.molar_mass_kgmol, temperature_K) : 0.f;
			if (isnan(dPdrho) || dPdrho < 1E-9f) { break; }
			float dr = fminf(fminf(max_dr, dPdrho * radius_km * radius_km / (G_km2_m_s2_Tg * .05f * mass_contained_within)), domain_size_km - radius_km + 1.f);

			radius_km += dr * .3333333333f;
			float dln_rho_guess = -(G_km2_m_s2_Tg * .001f) * mass_contained_within * dr / (radius_km * radius_km * dPdrho);
			float dmass_contained = 6.28318530718f * radius_km * radius_km * dr * rho * (1.f + expf(dln_rho_guess));
			ln_rho -= (G_km2_m_s2_Tg * .001f) * (mass_contained_within + dmass_contained * .75f) * dr / (radius_km * radius_km * dPdrho);
			mass_contained_within += dmass_contained;
			radius_km += dr * .6666666666f;
		}
		return mass_contained_within;
	}
public:
	hydrostatic_body_solver(float target_mass_Tg, float uniform_temperature_K, material_properties& mat) : mass_Tg(target_mass_Tg), temperature_K(uniform_temperature_K), radius_km(0), core_density_kgm3(0), mat(mat)
	{
		float lower_bound = 100.f, upper_bound = 100.f; int iters = 0;
		float left_value = total_mass_from_core_density(lower_bound) - target_mass_Tg, right_value = left_value;
		if (left_value > 0.f)
		{
			while (left_value > 0.f && iters < 100)
			{
				++iters;
				upper_bound = lower_bound;
				right_value = left_value;
				lower_bound *= .5f;
				left_value = total_mass_from_core_density(lower_bound) - target_mass_Tg;
			}
			if (abs(left_value) / target_mass_Tg < 1E-5f)
			{
				core_density_kgm3 = lower_bound;
				goto NOTHING;
			}
		}
		else {
			while (right_value < 0.f && iters < 100)
			{
				++iters;
				lower_bound = upper_bound;
				left_value = right_value;
				upper_bound *= 2.f;
				right_value = total_mass_from_core_density(upper_bound) - target_mass_Tg;
			}
			if (abs(right_value) / target_mass_Tg < 1E-5f)
			{
				core_density_kgm3 = upper_bound;
				goto NOTHING;
			}
		}
		float intermediate = (upper_bound * right_value - lower_bound * left_value) * .6f / (right_value - left_value) + (upper_bound + lower_bound) * .2f;
		float int_value = total_mass_from_core_density(intermediate) - target_mass_Tg;
		while (abs(int_value) / target_mass_Tg > 1E-5f && iters < 100)
		{
			++iters;
			if (int_value > 0.f) {
				upper_bound = intermediate;
				right_value = int_value;
			}
			else {
				lower_bound = intermediate;
				left_value = int_value;
			}
			intermediate = (upper_bound * right_value - lower_bound * left_value) * .6f / (right_value - left_value) + (upper_bound + lower_bound) * .2f;
			int_value = total_mass_from_core_density(intermediate) - target_mass_Tg;
		}
		core_density_kgm3 = intermediate;
	NOTHING:
	}
	stratification_data density_distribution(uint layers = 10u) const
	{
		stratification_data densities(layers, radius_km);
		if (layers < 4u)
		{
			densities.data.cpu_buffer_ptr[0] = mass_Tg * 0.23873241463f / (radius_km * radius_km * radius_km);
			for (uint i = 1; i < layers; i++)
				densities.data.cpu_buffer_ptr[i] = densities.data.cpu_buffer_ptr[0];
		}
		else {
			densities.data.cpu_buffer_ptr[0] = core_density_kgm3;
			float ln_rho = logf(core_density_kgm3);
			float mass_contained_within = 1E-10f;
			const float deltaR = radius_km / (layers - 1u);
			const float max_dr = cbrtf(1E-5f * mass_Tg / fmaxf(core_density_kgm3, mat.standard_density_kgm3));
			float radius = 1E-10f;

			for (uint i = 1; i < layers; i++)
			{
				for (float rad_step = 0.f; rad_step < deltaR * .9999f; )
				{
					const float rho = expf(ln_rho);
					float tolerance = fminf(max_dr, deltaR - rad_step);
					float dPdrho = mat.EOS_dp_drho_isothermal(rho, rho / mat.standard_density_kgm3, rho / mat.molar_mass_kgmol, temperature_K);
					float dr = fminf(tolerance, dPdrho * (radius + rad_step) * (radius + rad_step) / (G_km2_m_s2_Tg * .05f * mass_contained_within));
					dr = tolerance / ceilf(tolerance / dr);

					rad_step += dr * .3333333333f;
					float dln_rho_guess = -(G_km2_m_s2_Tg * .001f) * mass_contained_within * dr / ((radius + rad_step) * (radius + rad_step) * dPdrho);
					float dmass_contained = 6.28318530718f * (radius + rad_step) * (radius + rad_step) * dr * rho * (1.f + expf(dln_rho_guess));
					ln_rho -= (G_km2_m_s2_Tg * .001f) * (mass_contained_within + dmass_contained * .75f) * dr / ((radius + rad_step) * (radius + rad_step) * dPdrho);
					mass_contained_within += dmass_contained;
					rad_step += dr * .6666666666f;
				}
				radius += deltaR;
				densities.data.cpu_buffer_ptr[i] = expf(ln_rho);
			}
		}
		densities.data.copy_to_gpu();
		return densities;
	}
};

__global__ void __apply_density_strata(particle* particles, particle_kinematics* kinematics, const float* density_strata, 
	const float outer_radius, const float standard_density, const float3 center_pos, 
	const uint stratum_count, const uint particle_capacity, const uint offset_idx)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; } idx += offset_idx;
	if (!particles[idx].exists()) { return; }
	
	float radial_index = clamp(length(particles[idx].true_pos() - center_pos) / outer_radius, 0.f, 0.999999f) * (stratum_count - 1u);
	float density_scaling = lerp(density_strata[(uint)radial_index], density_strata[(uint)ceilf(radial_index)], fracf(radial_index)) / standard_density;
	kinematics[idx].mass_Tg *= density_scaling;
}

void __scale_for_densities(kinematic_simulation& simulation, const stratification_data& density_data, const uint start_idx, const uint count, const float reference_dens, const float3 center_pos)
{
	dim3 threads(min(count, 512u));
	dim3 blocks((uint)ceilf(count / (float)threads.x)); 
	__apply_density_strata<<<blocks, threads>>>(simulation.particles.buffer.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, density_data.data.gpu_buffer_ptr, 
		density_data.outer_radius, reference_dens, center_pos, density_data.data.dedicated_len, count, start_idx);
	cuda_sync();
}
std::vector<uint> initialize_thermodynamic_objects_hydrostatic_equilibrium(hydrogravitational_simulation& simulation, std::vector<initial_thermodynamic_object>& objects_mutated, bool center_of_mass_frame = true)
{
	std::vector<hydrostatic_body_solver> solved_bodies;
	for (uint i = 0u, s = objects_mutated.size(); i < s; i++)
	{
		if (objects_mutated[i].geometry_type != initial_kinematic_object::geometry::GEOM_SPHERE)
			continue;
		hydrostatic_body_solver solver = hydrostatic_body_solver(objects_mutated[i].total_mass_Tg, objects_mutated[i].temperature_K, 
																simulation.materials_cpu_copy.cpu_buffer_ptr[objects_mutated[i].material_index]);
		objects_mutated[i].dimensions[0] = solver.radius_km;
		solved_bodies.push_back(solver);
	}
	std::vector<uint> counts = initialize_thermodynamic_objects(simulation, objects_mutated, center_of_mass_frame);

	for (uint i = 0u, s = counts.size(), k = 0; i < s; k += counts[i], i++)
	{
		if (objects_mutated[i].geometry_type != initial_kinematic_object::geometry::GEOM_SPHERE)
			continue;
		stratification_data dat = solved_bodies[i].density_distribution((uint)ceilf(cbrtf(counts[i]) * 1.2f));
		__scale_for_densities(simulation, dat, k, counts[i], objects_mutated[i].total_mass_Tg / objects_mutated[i].volume_km3(), objects_mutated[i].center_pos_km);
		dat.destroy();
	}
	return counts;
}

#endif