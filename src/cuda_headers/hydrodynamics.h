#ifndef HYDRODYNAMICS_H
#define HYDRODYNAMICS_H
#include "gravitation.h"

__device__ constexpr float sph_monaghan_viscosity_alpha = .5f;
__device__ constexpr float sph_monaghan_viscosity_beta = 2.f;
__device__ constexpr float sph_cutoff_radius = .9f; // needs to be sufficiently large for accurate averaging. Less than 1.5
__device__ constexpr float ideal_gas_constant = 8.314f;
__device__ constexpr uint max_material_count = 16u;
__device__ constexpr float max_acceleration_factor_tick = 1E+5f; // Deletes particles with sudden accelerations too high to be reasonable.

// Separate into thermal and bulk pressure.
struct material_properties
{
	float limiting_heat_capacity_kJkgK;
	float thermal_scale_K;
	float molar_mass_kgmol;
	float standard_density_kgm3;
	float bulk_modulus_GPa;
	float stiffness_exponent;
	float2 padding;

	__device__ __host__ float temperature_K(float specific_energy_TJTg) const {
		specific_energy_TJTg /= limiting_heat_capacity_kJkgK;
		return .5f * (specific_energy_TJTg + sqrtf(4.f * thermal_scale_K + specific_energy_TJTg) * sqrtf(specific_energy_TJTg));
	}
	__device__ __host__ float specific_energy_TJTg(float temperature_K) const {
		return temperature_K * (temperature_K / (thermal_scale_K + temperature_K)) * limiting_heat_capacity_kJkgK;
	}
	__device__ __host__ float specific_heat_capacity_TJKTg(float temperature_K) const {

		float temp = temperature_K + thermal_scale_K;
		return temperature_K * (2.f * thermal_scale_K + temperature_K) * limiting_heat_capacity_kJkgK / (temp * temp);
	}
	__device__ __host__ float specific_entropy_TJKTg(float temperature_K) const {
		return limiting_heat_capacity_kJkgK * (temperature_K / (thermal_scale_K + temperature_K) + logf((thermal_scale_K + temperature_K) / thermal_scale_K));
	}
	__device__ __host__ float EOS_thermal_pressure_GPa(float volume_fraction, float number_density, float temperature_K) const
	{
		float temp = powf(volume_fraction, stiffness_exponent - 1.f);
		number_density *= (fabsf(volume_fraction - 1.f) < 1E-4f) ? (stiffness_exponent - 1.f) : (temp - 1.f) / (volume_fraction - 1.f);
		return temperature_K * (ideal_gas_constant * 1E-9f) * number_density;
	}
	__device__ __host__ float EOS_pressure_GPa(float volume_fraction, float number_density, float temperature_K) const
	{
		return fmaxf(0.f, bulk_modulus_GPa * (powf(volume_fraction, stiffness_exponent) - volume_fraction * volume_fraction) / (stiffness_exponent - 2.f)
			+ EOS_thermal_pressure_GPa(volume_fraction, number_density, temperature_K));
	}
	__device__ __host__ float EOS_speed_of_sound_kms(float density, float volume_fraction, float number_density, float temperature_K) const
	{
		float temp = powf(volume_fraction, stiffness_exponent - 1.f);
		float approximate_adiabatic_constant = 1.f + ideal_gas_constant / fmaxf(.00001f, molar_mass_kgmol * 1000.f * specific_heat_capacity_TJKTg(temperature_K));
		number_density *= temperature_K * approximate_adiabatic_constant * (ideal_gas_constant * 1E-6f) / density;
		float bulk_modulus_component = bulk_modulus_GPa * 1000.f * (temp * stiffness_exponent - volume_fraction * 2.f) * volume_fraction / (density * (stiffness_exponent - 2.f));
		float thermal_pressure = ((fabsf(volume_fraction - 1.f) < 1E-4f) ? (stiffness_exponent - 1.f) : (temp - 1.f) / (volume_fraction - 1.f)) * number_density, thermal_component;
		if (fabsf(volume_fraction - 1.f) < 1E-4f)
			thermal_component = (stiffness_exponent * (stiffness_exponent * .5f - 1.5f) + 1.f) * number_density;
		else
			thermal_component = number_density * (stiffness_exponent - 1.f) * temp / (volume_fraction - 1.f) - thermal_pressure / (1.f - 1.f / volume_fraction);
		return sqrtf(fmaxf(bulk_modulus_component + thermal_pressure + thermal_component, number_density));
	}
};
struct particle_thermodynamics
{
	uint material_index;
	float specific_thermal_energy_TJTg;

	__device__ __host__ particle_thermodynamics() : material_index(0u), specific_thermal_energy_TJTg(150.f) {}
};
struct SPH_variables
{
	float total_pressure_GPa;
	float thermal_pressure_GPa;
	float avg_density_kgm3;
	float speed_of_sound_kms;
	float density_change_rate;

	__device__ __host__ SPH_variables(const material_properties& mat_properties, const float density, const float volume_fraction, const float number_density,
	const float internal_energy, const float density_change_rate) : total_pressure_GPa(), thermal_pressure_GPa(), avg_density_kgm3(density), density_change_rate(density_change_rate), speed_of_sound_kms()
	{
		const float temperature = mat_properties.temperature_K(internal_energy);
		thermal_pressure_GPa = mat_properties.EOS_thermal_pressure_GPa(volume_fraction, number_density, temperature); // This is without interparticle potentials.
		total_pressure_GPa = mat_properties.EOS_pressure_GPa(volume_fraction, number_density, temperature);
		speed_of_sound_kms = mat_properties.EOS_speed_of_sound_kms(density, volume_fraction, number_density, temperature);
	}
	__device__ __host__ SPH_variables() : total_pressure_GPa(), avg_density_kgm3() {}
};

/////////////////////////////////////////
//// Smoothed Particle Hydrodynamics ////
/////////////////////////////////////////

__constant__ __device__ material_properties materials[max_material_count]; // Resides in constant memory. Needs to be sufficiently small.
static_assert(sizeof(material_properties) * max_material_count < 4096u, "Constant memory insufficient!");

__device__ constexpr float __sq_dist_cutoff = sph_cutoff_radius * size_grid_cell_km * sph_cutoff_radius * size_grid_cell_km;
inline __device__ __host__ float ___radius_factor(float this_radius, float other_radius)
{
	return fminf(this_radius * this_radius + other_radius * other_radius, __sq_dist_cutoff * 0.22222222222f);
}
inline __device__ __host__ float ___spline_kernel(float sq_displacement_km2, float radius_factor)
{
	sq_displacement_km2 /= radius_factor * 4.5f; float displacement_factor = sqrtf(sq_displacement_km2);
	return fmaxf(0.f, .2f - sq_displacement_km2 * (1.f - displacement_factor) - sq_displacement_km2
		* sq_displacement_km2 * displacement_factor * 0.2f) / (radius_factor * sqrtf(radius_factor));
}
inline __device__ __host__ float ___spline_kernel_grad_factor(float sq_displacement_km2, float radius_factor)
{
	sq_displacement_km2 /= radius_factor * 4.5f; if (sq_displacement_km2 > 1.f) { return 0.f; } float displacement_factor = sqrtf(sq_displacement_km2);
	return (2.f - displacement_factor * (3.f - sq_displacement_km2)) * 0.22222222222f / (radius_factor * radius_factor * sqrtf(radius_factor));
}

__global__ void __average_SPH_quantities(SPH_variables* average, const uint* cell_bounds, const particle_thermodynamics* thermodynamics, 
											const particle_kinematics* kinematics, const particle* particles, const uint particle_capacity)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	if (!particles[idx].exists()) { return; }
	const float3 this_pos = particles[idx].true_pos();
	const float3 this_vel = kinematics[idx].velocity_kms;
	const uint morton_index = particles[idx].morton_index();
	const float this_radius_km = kinematics[idx].radius_km;

	morton_cell_iterator iter = morton_cell_iterator(morton_index);
	float average_number_density_molm3 = 0.f, average_volume_fraction = 0.f, average_density_kgm3 = 0.f, density_rate_change_kgm3s = 0.f;

	FOREACH(uint, loop_morton, iter)
		for (uint i = __read_start_idx(cell_bounds, loop_morton), end = __read_end_idx(cell_bounds, loop_morton); i < end; i++)
		{
			float3 separation = particles[i].true_pos() - this_pos;
			if (dot(separation, separation) >= __sq_dist_cutoff) { continue; }

			float radius_factor = ___radius_factor(kinematics[i].radius_km, this_radius_km);
			const float density_fraction = ___spline_kernel(dot(separation, separation), radius_factor) * kinematics[i].mass_Tg;
			uint other_mat_idx = thermodynamics[i].material_index;

			average_density_kgm3 += density_fraction;
			average_number_density_molm3 += density_fraction / materials[other_mat_idx].molar_mass_kgmol;
			average_volume_fraction += density_fraction / materials[other_mat_idx].standard_density_kgm3;
			density_rate_change_kgm3s += dot(separation, this_vel - kinematics[i].velocity_kms) * density_fraction / radius_factor;
		}

	average[idx] = SPH_variables(materials[thermodynamics[idx].material_index], average_density_kgm3, average_volume_fraction,
		average_number_density_molm3, thermodynamics[idx].specific_thermal_energy_TJTg, density_rate_change_kgm3s);
}
__global__ void __apply_SPH_forces(const SPH_variables* average, const uint* cell_bounds, particle_thermodynamics* thermodynamics,
									particle_kinematics* kinematics, particle* particles, const uint particle_capacity, const float timestep)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }
	if (!particles[idx].exists()) { return; }

	const float3 this_pos = particles[idx].true_pos();
	const float3 this_vel = kinematics[idx].velocity_kms;
	const float this_radius_km = kinematics[idx].radius_km;
	const SPH_variables this_data = average[idx];

	morton_cell_iterator iter = morton_cell_iterator(particles[idx].morton_index());
	float3 hydrodynamic_acceleration_ms2 = make_float3(0.f);
	float specific_internal_energy_change_TJTg = 0.f;

	FOREACH(uint, loop_morton, iter)
		for (uint i = __read_start_idx(cell_bounds, loop_morton), end = __read_end_idx(cell_bounds, loop_morton); i < end; i++)
		{
			float3 displacement = this_pos - particles[i].true_pos();
			const float sq_dst = dot(displacement, displacement);
			if (sq_dst >= __sq_dist_cutoff) { continue; }

			const float3 relative_velocity = kinematics[i].velocity_kms - this_vel;
			float radius_factor = ___radius_factor(kinematics[i].radius_km, this_radius_km);
			float monaghan_viscosity_parameter = fmaxf(0.f, dot(relative_velocity, displacement)) * sqrtf(radius_factor) / (sq_dst + radius_factor * .0001f);
			const float other_density = average[i].avg_density_kgm3;

			displacement *= ___spline_kernel_grad_factor(sq_dst, radius_factor) * kinematics[i].mass_Tg;
			monaghan_viscosity_parameter *= (sph_monaghan_viscosity_alpha * (this_data.speed_of_sound_kms + average[i].speed_of_sound_kms) * 1000.f 
				+ monaghan_viscosity_parameter * (sph_monaghan_viscosity_beta * 2000.f)) / (other_density + this_data.avg_density_kgm3); // artificial viscosity; needs factor of 1000 for units to work out
			
			float thermal_pressure_mul = (average[i].thermal_pressure_GPa / (other_density * other_density)
				+ this_data.thermal_pressure_GPa / (this_data.avg_density_kgm3 * this_data.avg_density_kgm3)) * 1E+6f + monaghan_viscosity_parameter;
			float total_pressure_mul = (average[i].total_pressure_GPa / (other_density * other_density) 
				+ this_data.total_pressure_GPa / (this_data.avg_density_kgm3 * this_data.avg_density_kgm3)) * 1E+6f + monaghan_viscosity_parameter;

			hydrodynamic_acceleration_ms2 += displacement * total_pressure_mul;
			specific_internal_energy_change_TJTg += dot(displacement, relative_velocity) * thermal_pressure_mul; // discounts potential energy from interparticle potential.
		}

	kinematics[idx].acceleration_ms2 += hydrodynamic_acceleration_ms2;
	thermodynamics[idx].specific_thermal_energy_TJTg = fmaxf(thermodynamics[idx].specific_thermal_energy_TJTg + .5f * specific_internal_energy_change_TJTg * timestep, .01f);
}
__global__ void __step_particle_data(const SPH_variables* average, const particle_thermodynamics* thermodynamics, particle_kinematics* kinematics, particle* particles, const uint particle_capacity, const float timestep)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (idx >= particle_capacity) { return; }
	const float standard_density = materials[thermodynamics[idx].material_index].standard_density_kgm3;
	const float minimum_radius_permissible = cbrtf(kinematics[idx].mass_Tg / standard_density) * 0.062035049089f;
	const float actual_density = average[idx].avg_density_kgm3;

	const float factor = expf(-average[idx].density_change_rate * timestep * .33333333333f / actual_density);
	kinematics[idx].multiply_radius(factor, minimum_radius_permissible, sph_cutoff_radius * size_grid_cell_km * 0.47140452078f);
	if (!(length(kinematics[idx].acceleration_ms2) * timestep * timestep < max_acceleration_factor_tick))
		particles[idx].set_existence(false);
}
__global__ void __init_thermodynamics(particle_thermodynamics* thermodynamics, const uint start_idx, const uint length, const particle_thermodynamics thr)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= length) { return; }

	thermodynamics[idx + start_idx] = thr;
}

//////////////////////////////////
////	  Main Structures	  ////
//////////////////////////////////

struct hydrodynamics_simulation : virtual public kinematic_simulation
{
	particle_data_buffer<particle_thermodynamics> thermodynamic_data;
	smart_gpu_buffer<SPH_variables> smoothed_particle_hydrodynamics;
	smart_cpu_buffer<material_properties> materials_cpu_copy;

	hydrodynamics_simulation(size_t allocation_particles) : kinematic_simulation(allocation_particles), thermodynamic_data(allocation_particles), smoothed_particle_hydrodynamics(allocation_particles), materials_cpu_copy(max_material_count)
	{	
		dim3 threads(allocation_particles > 512u ? 512u : allocation_particles);
		dim3 blocks((uint)ceilf(allocation_particles / (float)threads.x));

		__set_empty<<<blocks, threads>>>(thermodynamic_data.buffer.gpu_buffer_ptr, allocation_particles);
	}
	void set_thermodynamics(uint start_index, uint number_of_particles, uint material_index = 0u, float temperature_K = 298.f)
	{
		dim3 threads(number_of_particles > 512u ? 512u : number_of_particles);
		dim3 blocks((uint)ceilf(number_of_particles / (float)threads.x));

		particle_thermodynamics temp = particle_thermodynamics();
		material_properties& properties = materials_cpu_copy.cpu_buffer_ptr[material_index];
		if (properties.limiting_heat_capacity_kJkgK == 0.f || properties.molar_mass_kgmol == 0.f) { throw std::logic_error("Material Properties invalid."); }
		temp.material_index = material_index; temp.specific_thermal_energy_TJTg = fmaxf(0.01f, properties.specific_energy_TJTg(temperature_K));
		__init_thermodynamics<<<blocks, threads>>>(thermodynamic_data.buffer.gpu_buffer_ptr, start_index, number_of_particles, temp);
	}
	void copy_materials_to_gpu() const
	{
		cudaMemcpyToSymbol(materials, materials_cpu_copy.cpu_buffer_ptr, materials_cpu_copy.total_size());
	}
	void compute_sph_quantities() {
		dim3 threads(particle_capacity > 512u ? 512u : particle_capacity);
		dim3 blocks((uint)ceilf(particle_capacity / (float)threads.x));
		
		__average_SPH_quantities<<<blocks, threads>>>(smoothed_particle_hydrodynamics.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr, 
			thermodynamic_data.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particle_capacity);
	}
	void apply_thermodynamic_timestep(const float timestep, bool apply_heat = true)
	{
		dim3 threads(particle_capacity > 512u ? 512u : particle_capacity);
		dim3 blocks((uint)ceilf(particle_capacity / (float)threads.x));
		
		__apply_SPH_forces<<<blocks, threads>>>(smoothed_particle_hydrodynamics.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr,
			thermodynamic_data.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particle_capacity, timestep * apply_heat);

		__step_particle_data<<<blocks, threads>>>(smoothed_particle_hydrodynamics.gpu_buffer_ptr, thermodynamic_data.buffer.gpu_buffer_ptr, 
												  kinematic_data.buffer.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particle_capacity, timestep);
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
		compute_sph_quantities();
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


struct initial_thermodynamic_object : initial_kinematic_object
{
	float temperature_K;
	uint material_index;

	initial_thermodynamic_object()
	{

	}

	initial_thermodynamic_object(geometry geometry_type, std::vector<float> dimensions, float total_mass_Tg,
		float3 center_pos_km = make_float3(domain_size_km * .5f), float3 velocity_kms = make_float3(0.f), float3 angular_velocity_rads = make_float3(0.f), 
		float temperature_K = 298.f, uint material_index = 0u) : initial_kinematic_object(geometry_type, dimensions, total_mass_Tg, center_pos_km, velocity_kms, angular_velocity_rads), temperature_K(temperature_K), material_index(material_index)
	{
	}
};
std::vector<uint> initialize_thermodynamic_objects(hydrodynamics_simulation& simulation, const std::vector<initial_thermodynamic_object>& objects, bool center_of_mass_frame = true)
{
	std::vector<initial_kinematic_object> to_kin(objects.size());
	for (uint i = 0u, s = objects.size(); i < s; i++) { to_kin[i] = objects[i]; }
	std::vector<uint> counts = initialize_kinematic_objects(simulation, to_kin, center_of_mass_frame);

	for (uint i = 0u, s = counts.size(), k = 0; i < s; k += counts[i], i++)
	{
		const initial_thermodynamic_object& obj = objects[i];
		simulation.set_thermodynamics(k, counts[i], obj.material_index, obj.temperature_K);
	}
	return counts;
}

#endif