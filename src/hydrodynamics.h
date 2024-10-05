#ifndef HYDRODYNAMICS_H
#define HYDRODYNAMICS_H
#include "gravitation.h"

__device__ constexpr float sph_monaghan_viscosity_alpha = 1.f;
__device__ constexpr float sph_monaghan_viscosity_beta = 2.f;
__device__ constexpr float sph_cutoff_radius = 1.f;
__device__ constexpr float ideal_gas_constant = 8.314f;
__device__ constexpr uint max_material_count = 16u;

struct material_properties
{
	float limiting_heat_capacity_kJkgK;
	float thermal_scale_K;
	float standard_volume_m3mol;
	float standard_density_kgm3;
	float bulk_modulus_GPa;
	float3 padding;

	__device__ __host__ float temperature_K(float specific_energy_TJTg) const {
		specific_energy_TJTg /= limiting_heat_capacity_kJkgK;
		return .5f * (specific_energy_TJTg + sqrtf(4.f * specific_energy_TJTg * thermal_scale_K + specific_energy_TJTg * specific_energy_TJTg));
	}
	__device__ __host__ float specific_energy_TJTg(float temperature_K) const {
		return temperature_K * temperature_K / (thermal_scale_K + temperature_K) * limiting_heat_capacity_kJkgK;
	}
	__device__ __host__ float specific_heat_capacity_TJKTg(float temperature_K) const {

		float temp = temperature_K + thermal_scale_K;
		return temperature_K * (2.f * thermal_scale_K + temperature_K) * limiting_heat_capacity_kJkgK / (temp * temp);
	}
	__device__ __host__ float specific_entropy_TJKTg(float temperature_K) const {
		return limiting_heat_capacity_kJkgK * (temperature_K / (thermal_scale_K + temperature_K) + logf((thermal_scale_K + temperature_K) / thermal_scale_K));
	}
	__device__ __host__ float EOS_pressure_GPa(float volume_fraction, float number_density, float temperature_K)
	{
		float t = 1.f + volume_fraction, f = volume_fraction * volume_fraction; 
		t = 1.f + volume_fraction * t; t = number_density * (1.f + volume_fraction * t);
		return fmaxf(0.f, 0.3333333333333f * bulk_modulus_GPa * (f * f - volume_fraction) * volume_fraction + temperature_K * (ideal_gas_constant * 1E-9f) * t);
	}
};
struct particle_thermodynamics
{
	uint material_index;
	float specific_internal_energy_TJTg;
	float expansion_rate_ratio;
	float padding;

	__device__ __host__ void change_internal_energy(float change) {
		specific_internal_energy_TJTg = fmaxf(specific_internal_energy_TJTg + change, 0.f);
	}
	__device__ __host__ particle_thermodynamics() : material_index(0u), specific_internal_energy_TJTg(208.6f), expansion_rate_ratio(0.f), padding() {}
};
struct SPH_variables
{
	float pressure_GPa;
	float avg_density_kgm3;

	__device__ __host__ SPH_variables() : pressure_GPa(), avg_density_kgm3() {}
};

__constant__ __device__ material_properties materials[max_material_count];
inline __device__ __host__ float ___sph_kernel(float sq_displacement_km2, float this_radius_km, float other_radius_km)
{
	this_radius_km = this_radius_km * this_radius_km + other_radius_km * other_radius_km;
	return expf(-0.5f * sq_displacement_km2 / this_radius_km) * 0.06349363593424097f / (this_radius_km * sqrtf(this_radius_km));
}

__device__ constexpr float __dist_cutoff = sph_cutoff_radius * size_grid_cell_km;
__device__ constexpr float __sq_dist_cutoff = __dist_cutoff * __dist_cutoff;
inline __device__ __host__ float ___sph_kernel_w_cutoff(float sq_displacement_km2, float this_radius_km, float other_radius_km)
{
	float temp = ___sph_kernel(__sq_dist_cutoff, this_radius_km, other_radius_km);
	float temp2 = this_radius_km * this_radius_km + other_radius_km * other_radius_km;
	return fmaxf(0.f, (___sph_kernel(sq_displacement_km2, this_radius_km, other_radius_km) - temp) 
	/ (erff(__dist_cutoff * 0.70710678118f * rsqrtf(temp2)) - 4.18879020479f * __dist_cutoff * (3.f * temp2 + __sq_dist_cutoff) * temp));
}

__global__ void __average_SPH_quantities(SPH_variables* average, const uint* cell_bounds, const particle_thermodynamics* thermodynamics, 
											const particle_kinematics* kinematics, const particle* particles, const uint particle_capacity)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	if (!particles[idx].exists()) { return; }
	const float3 this_pos = particles[idx].true_pos();
	const uint morton_index = particles[idx].morton_index();
	const float this_radius_km = kinematics[idx].radius_km;

	morton_cell_iterator iter = morton_cell_iterator(morton_index);
	float average_number_density_molm3 = 0.f, average_volume_fraction = 0.f, average_density_kgm3 = 0.f;

	FOREACH(uint, loop_morton, iter)
		for (uint i = __read_start_idx(cell_bounds, loop_morton), end = __read_end_idx(cell_bounds, loop_morton); i < end; i++)
		{
			float3 separation = particles[i].true_pos() - this_pos;
			if (dot(separation, separation) >= __sq_dist_cutoff) { continue; }

			float density_fraction = ___sph_kernel_w_cutoff(dot(separation, separation), this_radius_km, kinematics[i].radius_km) * kinematics[i].mass_Tg;
			uint other_mat_idx = thermodynamics[i].material_index;

			average_density_kgm3 += density_fraction;
			average_number_density_molm3 += density_fraction / (materials[other_mat_idx].standard_volume_m3mol * materials[other_mat_idx].standard_density_kgm3);
			average_volume_fraction += density_fraction / materials[other_mat_idx].standard_density_kgm3;
		}

	const uint material_idx = thermodynamics[idx].material_index;
	const float temperature_K = materials[material_idx].temperature_K(thermodynamics[idx].specific_internal_energy_TJTg);
	SPH_variables result; result.avg_density_kgm3 = average_density_kgm3;
	result.pressure_GPa = materials[material_idx].EOS_pressure_GPa(average_volume_fraction, average_number_density_molm3, temperature_K);
	average[idx] = result;
}

__global__ void __apply_SPH_forces(const SPH_variables* average, const uint* cell_bounds, particle_thermodynamics* thermodynamics,
									particle_kinematics* kinematics, const particle* particles, const uint particle_capacity, const float timestep)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	if (!particles[idx].exists()) { return; }

	const float3 this_pos = particles[idx].true_pos();
	const float3 this_vel = kinematics[idx].velocity_kms;
	const uint morton_index = particles[idx].morton_index();
	const float this_radius_km = kinematics[idx].radius_km;
	const SPH_variables this_data = average[idx];

	morton_cell_iterator iter = morton_cell_iterator(morton_index);
	float3 hydrodynamic_acceleration_ms2 = make_float3(0.f);
	float specific_internal_energy_change_TJTg = 0.f, density_rate_change_kgm3s = 0.f;

	FOREACH(uint, loop_morton, iter)
		for (uint i = __read_start_idx(cell_bounds, loop_morton), end = __read_end_idx(cell_bounds, loop_morton); i < end; i++)
		{
			float3 displacement = this_pos - particles[i].true_pos();
			const float sq_dst = dot(displacement, displacement);
			if (sq_dst >= __sq_dist_cutoff) { continue; }

			const float3 relative_velocity = kinematics[i].velocity_kms - this_vel;
			float radius_factor = kinematics[i].radius_km; radius_factor *= radius_factor; radius_factor += this_radius_km * this_radius_km;
			const float monaghan_viscosity_parameter = fmaxf(0.f, dot(relative_velocity, displacement)) * sqrtf(radius_factor) / (sq_dst + radius_factor * .01f);
			const float other_density = average[i].avg_density_kgm3;

			displacement *= ___sph_kernel_w_cutoff(sq_dst, this_radius_km, kinematics[i].radius_km) * kinematics[i].mass_Tg / radius_factor;
			density_rate_change_kgm3s -= dot(displacement, relative_velocity);

			displacement *= ((average[i].pressure_GPa + this_data.pressure_GPa) * 1E+6f / (this_data.avg_density_kgm3 * other_density) // pressure
			+ monaghan_viscosity_parameter * (sph_monaghan_viscosity_alpha + monaghan_viscosity_parameter * sph_monaghan_viscosity_beta) * 2000.f / (other_density + this_data.avg_density_kgm3)); // artificial viscosity; needs factor of 1000 for units to work out
			
			hydrodynamic_acceleration_ms2 += displacement;
			specific_internal_energy_change_TJTg += dot(displacement, relative_velocity);
		}

	kinematics[idx].acceleration_ms2 += hydrodynamic_acceleration_ms2;
	thermodynamics[idx].change_internal_energy(0.5f * specific_internal_energy_change_TJTg * timestep);
	thermodynamics[idx].expansion_rate_ratio = density_rate_change_kgm3s * timestep * .3f / this_data.avg_density_kgm3;
}

__global__ void __step_particle_radius(const particle_thermodynamics* thermodynamics, particle_kinematics* kinematics, const uint particle_capacity)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (idx >= particle_capacity) { return; }

	const float min_allowable_radius = cbrtf(kinematics[idx].mass_Tg / materials[thermodynamics[idx].material_index].standard_density_kgm3) * 0.49f;
	kinematics[idx].multiply_radius(expf(thermodynamics[idx].expansion_rate_ratio), min_allowable_radius, sph_cutoff_radius * size_grid_cell_km * .4f);
}

struct hydrodynamics_simulation : virtual public kinematic_simulation
{
	particle_data_buffer<particle_thermodynamics> thermodynamic_data;
	smart_gpu_buffer<SPH_variables> smoothed_particle_hydrodynamics;

	hydrodynamics_simulation(size_t allocation_particles) : kinematic_simulation(allocation_particles), thermodynamic_data(allocation_particles), smoothed_particle_hydrodynamics(allocation_particles)
	{	
		dim3 threads(allocation_particles > 512u ? 512u : allocation_particles);
		dim3 blocks((uint)ceilf(allocation_particles / (float)threads.x));

		__set_empty<<<blocks, threads>>>(thermodynamic_data.buffer.gpu_buffer_ptr, allocation_particles);
	}
	void copy_materials_to_gpu(smart_cpu_buffer<material_properties>& mats) const
	{
		cudaMemcpyToSymbol(materials, mats.cpu_buffer_ptr, min(mats.dedicated_len, max_material_count) * sizeof(material_properties));
	}
	void apply_thermodynamic_timestep(const float timestep)
	{
		dim3 threads(particle_capacity > 512u ? 512u : particle_capacity);
		dim3 blocks((uint)ceilf(particle_capacity / (float)threads.x));
		
		__average_SPH_quantities<<<blocks, threads>>>(smoothed_particle_hydrodynamics.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr, 
			thermodynamic_data.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particle_capacity);

		__apply_SPH_forces<<<blocks, threads>>>(smoothed_particle_hydrodynamics.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr,
			thermodynamic_data.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particle_capacity, timestep);

		__step_particle_radius<<<blocks, threads>>>(thermodynamic_data.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particle_capacity);
	}
	virtual void counting_sort_transfers(const smart_gpu_buffer<uint>& cell_bounds, const smart_gpu_buffer<particle>& targets) override {
		counting_sort_data_transfer(cell_bounds, targets, thermodynamic_data);
	}
};

struct hydrogravitational_simulation : virtual public hydrodynamics_simulation, virtual public gravitational_simulation
{
	hydrogravitational_simulation(size_t allocation_particles) : hydrodynamics_simulation(allocation_particles), gravitational_simulation(allocation_particles), kinematic_simulation(allocation_particles) {}
	void apply_complete_timestep(const float timestep, float recenter_strength = 1.f)
	{
		sort_spatially();
		generate_gravitational_data();
		apply_gravitation();
		apply_thermodynamic_timestep(timestep);

		if (recenter_strength > 0.f)
			apply_kinematics_recenter(timestep, recenter_strength);
		else
			apply_kinematics(timestep);
	}
	virtual void counting_sort_transfers(const smart_gpu_buffer<uint>& cell_bounds, const smart_gpu_buffer<particle>& targets) override {
		gravitational_simulation::counting_sort_transfers(cell_bounds, targets);
		hydrodynamics_simulation::counting_sort_transfers(cell_bounds, targets);
	}
};

#endif