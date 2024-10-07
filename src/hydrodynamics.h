#ifndef HYDRODYNAMICS_H
#define HYDRODYNAMICS_H
#include "gravitation.h"

__device__ constexpr float sph_monaghan_viscosity_alpha = 1.f;
__device__ constexpr float sph_monaghan_viscosity_beta = 2.f;
__device__ constexpr float sph_cutoff_radius = .9f; // needs to be sufficiently large for accurate averaging. Less than 1.5
__device__ constexpr float ideal_gas_constant = 8.314f;
__device__ constexpr uint max_material_count = 16u;

//////////////////////////////////
////	     Stability		  ////
//////////////////////////////////

__device__ constexpr float max_heat_tick_ratio = .1f;  // Maximum percentage of specific internal energy change per tick; remaining heat added in later ticks. 
													   // Used to prevent instabilities in thermodynamic calculations. Larger = more accurate but more unstable.
__device__ constexpr float max_heat_tick_TJTg = 100.f; // Same as above; adds constant to clamp.
__device__ constexpr float max_acceleration_factor_tick = 1E+5f; // Deletes particles with sudden accelerations too high to be reasonable.


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
	__device__ __host__ float EOS_pressure_GPa(float volume_fraction, float number_density, float temperature_K)
	{
		float temp = powf(volume_fraction, stiffness_exponent - 1.f);
		number_density *= (fabsf(volume_fraction - 1.f) < 1E-4f) ? (stiffness_exponent - 1.f) : (temp - 1.f) / (volume_fraction - 1.f);
		return fmaxf(0.f, bulk_modulus_GPa * (temp - volume_fraction) * volume_fraction / (stiffness_exponent - 2.f) + temperature_K * (ideal_gas_constant * 1E-9f) * number_density);
	}
};
struct particle_thermodynamics
{
	uint material_index;
	float specific_internal_energy_TJTg;
	float density_change_rate;
	float turbulent_kinetic_energy; // Numerical stabiliser akin to turbulent KE.

	__device__ __host__ void change_internal_energy(float energy_change) {
		energy_change *= .5f;
		float transfer = clamp(turbulent_kinetic_energy += energy_change,
			            fmaxf(-sqrtf(specific_internal_energy_TJTg * max_heat_tick_ratio) - max_heat_tick_TJTg, .01f - specific_internal_energy_TJTg), 
			            sqrtf(specific_internal_energy_TJTg * max_heat_tick_ratio) + max_heat_tick_TJTg); // viscosity approximately proportional to sqrt internal energy.
		specific_internal_energy_TJTg += transfer;
		turbulent_kinetic_energy += energy_change - transfer;
	}
	__device__ __host__ particle_thermodynamics() : material_index(0u), specific_internal_energy_TJTg(150.f), density_change_rate(0.f), turbulent_kinetic_energy(0.f) {}
};
struct SPH_variables
{
	float pressure_GPa;
	float avg_density_kgm3;

	__device__ __host__ SPH_variables() : pressure_GPa(), avg_density_kgm3() {}
};

/////////////////////////////////////////
//// Smoothed Particle Hydrodynamics ////
/////////////////////////////////////////

__constant__ __device__ material_properties materials[max_material_count];
inline __device__ __host__ float ___sph_kernel(float sq_displacement_km2, float this_radius_km, float other_radius_km)
{
	this_radius_km = this_radius_km * this_radius_km + other_radius_km * other_radius_km;
	return expf(-0.5f * sq_displacement_km2 / this_radius_km) * 0.06349363593424097f / (this_radius_km * sqrtf(this_radius_km));
}
__device__ constexpr float __sq_dist_cutoff = sph_cutoff_radius * size_grid_cell_km * sph_cutoff_radius * size_grid_cell_km;


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

			float density_fraction = ___sph_kernel(dot(separation, separation), this_radius_km, kinematics[i].radius_km) * kinematics[i].mass_Tg;
			uint other_mat_idx = thermodynamics[i].material_index;

			average_density_kgm3 += density_fraction;
			average_number_density_molm3 += density_fraction / materials[other_mat_idx].molar_mass_kgmol;
			average_volume_fraction += density_fraction / materials[other_mat_idx].standard_density_kgm3;
		}

	const uint material_idx = thermodynamics[idx].material_index;
	SPH_variables result; result.avg_density_kgm3 = average_density_kgm3;
	result.pressure_GPa = materials[material_idx].EOS_pressure_GPa(average_volume_fraction, average_number_density_molm3,
						  materials[material_idx].temperature_K(thermodynamics[idx].specific_internal_energy_TJTg));
	average[idx] = result;
}
__global__ void __apply_SPH_forces(const SPH_variables* average, const uint* cell_bounds, particle_thermodynamics* thermodynamics,
									particle_kinematics* kinematics, particle* particles, const uint particle_capacity, const float timestep)
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

			displacement *= ___sph_kernel(sq_dst, this_radius_km, kinematics[i].radius_km) * kinematics[i].mass_Tg / radius_factor;
			density_rate_change_kgm3s -= dot(displacement, relative_velocity);

			displacement *= ((average[i].pressure_GPa / (other_density * other_density) + this_data.pressure_GPa / (this_data.avg_density_kgm3 * this_data.avg_density_kgm3)) * 1E+6f // pressure
			+ monaghan_viscosity_parameter * (sph_monaghan_viscosity_alpha + monaghan_viscosity_parameter * sph_monaghan_viscosity_beta) * 2000.f / (other_density + this_data.avg_density_kgm3)); // artificial viscosity; needs factor of 1000 for units to work out
			
			hydrodynamic_acceleration_ms2 += displacement;
			specific_internal_energy_change_TJTg += dot(displacement, relative_velocity);
		}

	kinematics[idx].acceleration_ms2 += hydrodynamic_acceleration_ms2;
	thermodynamics[idx].change_internal_energy(0.5f * specific_internal_energy_change_TJTg * timestep);
	thermodynamics[idx].density_change_rate = density_rate_change_kgm3s;
}
__global__ void __step_particle_data(const SPH_variables* average, const particle_thermodynamics* thermodynamics, particle_kinematics* kinematics, particle* particles, const uint particle_capacity, const float timestep)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (idx >= particle_capacity) { return; }
	const float min_allowable_radius = cbrtf(kinematics[idx].mass_Tg / materials[thermodynamics[idx].material_index].standard_density_kgm3) * 0.1f;
	const float factor = expf(thermodynamics[idx].density_change_rate * timestep * .33333333333f / average[idx].avg_density_kgm3);
	kinematics[idx].multiply_radius(factor, min_allowable_radius, sph_cutoff_radius * size_grid_cell_km * .42f);
	if (length(kinematics[idx].acceleration_ms2) * timestep * timestep > max_acceleration_factor_tick)
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
		temp.material_index = material_index; temp.specific_internal_energy_TJTg = properties.specific_energy_TJTg(temperature_K);
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
	void apply_thermodynamic_timestep(const float timestep)
	{
		dim3 threads(particle_capacity > 512u ? 512u : particle_capacity);
		dim3 blocks((uint)ceilf(particle_capacity / (float)threads.x));
		
		__apply_SPH_forces<<<blocks, threads>>>(smoothed_particle_hydrodynamics.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr,
			thermodynamic_data.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particle_capacity, timestep);

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


//////////////////////////////////
////	    XSPH Variant 	  ////
//////////////////////////////////

__device__ constexpr float XSPH_x_factor_mul = .75f;
__global__ void __xsph_compute_x_factor(float3* x_factor, const SPH_variables* average, const uint* cell_bounds,
	const particle_kinematics* kinematics, const particle* particles, const uint particle_capacity)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	float3 x = make_float3(0.f);
	if (particles[idx].exists()) {
		const float3 this_pos = particles[idx].true_pos();
		const float3 this_vel = kinematics[idx].velocity_kms;
		const uint morton_index = particles[idx].morton_index();
		const float this_radius_km = kinematics[idx].radius_km;
		const float this_density = average[idx].avg_density_kgm3;

		morton_cell_iterator iter = morton_cell_iterator(morton_index);

		FOREACH(uint, loop_morton, iter)
			for (uint i = __read_start_idx(cell_bounds, loop_morton), end = __read_end_idx(cell_bounds, loop_morton); i < end; i++)
			{
				float3 separation = particles[i].true_pos() - this_pos;
				if (dot(separation, separation) >= __sq_dist_cutoff) { continue; }

				x += (___sph_kernel(dot(separation, separation), this_radius_km, kinematics[i].radius_km) * kinematics[i].mass_Tg * 2.f / (this_density + average[i].avg_density_kgm3)) * (kinematics[i].velocity_kms - this_vel);
			}
	}
	x_factor[idx] = x;
}
__global__ void __xsph_apply_x_factor(const float3* x_factor, particle_kinematics* kinematics, const uint particle_capacity, const float multiplier)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	kinematics[idx].velocity_kms += x_factor[idx] * multiplier;
}

struct XSPH_variant
{
	smart_gpu_buffer<float3> x_factor;
	XSPH_variant(uint allocation) : x_factor(allocation) {}

	void apply_XSPH_variant(hydrodynamics_simulation& simulation, const float timestep)
	{
		dim3 threads(simulation.particle_capacity > 512u ? 512u : simulation.particle_capacity);
		dim3 blocks((uint)ceilf(simulation.particle_capacity / (float)threads.x));

		simulation.sort_spatially();
		simulation.compute_sph_quantities();
		__xsph_compute_x_factor<<<blocks, threads>>>(x_factor.gpu_buffer_ptr, simulation.smoothed_particle_hydrodynamics.gpu_buffer_ptr, simulation.cell_bounds.gpu_buffer_ptr,
			simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particles.buffer.gpu_buffer_ptr, simulation.particle_capacity);
		__xsph_apply_x_factor<<<blocks, threads>>>(x_factor.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particle_capacity, XSPH_x_factor_mul);
		simulation.apply_thermodynamic_timestep(timestep);
		simulation.apply_kinematics(timestep);
		__xsph_apply_x_factor<<<blocks, threads>>>(x_factor.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particle_capacity, -XSPH_x_factor_mul);
	}
	void apply_XSPH_variant(hydrogravitational_simulation& simulation, const float timestep, float recenter_strength = 1.f)
	{
		dim3 threads(simulation.particle_capacity > 512u ? 512u : simulation.particle_capacity);
		dim3 blocks((uint)ceilf(simulation.particle_capacity / (float)threads.x));

		simulation.sort_spatially();
		simulation.generate_gravitational_data();
		simulation.apply_gravitation();
		simulation.compute_sph_quantities();
		__xsph_compute_x_factor<<<blocks, threads>>>(x_factor.gpu_buffer_ptr, simulation.smoothed_particle_hydrodynamics.gpu_buffer_ptr, simulation.cell_bounds.gpu_buffer_ptr,
			simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particles.buffer.gpu_buffer_ptr, simulation.particle_capacity);
		__xsph_apply_x_factor<<<blocks, threads>>>(x_factor.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particle_capacity, XSPH_x_factor_mul);
		simulation.apply_thermodynamic_timestep(timestep);
		if (recenter_strength > 0.f)
			simulation.apply_kinematics_recenter(timestep, recenter_strength);
		else
			simulation.apply_kinematics(timestep);
		__xsph_apply_x_factor<<<blocks, threads>>>(x_factor.gpu_buffer_ptr, simulation.kinematic_data.buffer.gpu_buffer_ptr, simulation.particle_capacity, -XSPH_x_factor_mul);
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