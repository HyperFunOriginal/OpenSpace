#ifndef KINEMATICS_H
#define KINEMATICS_H
#include "simulation_helper.h"
#include "cum_sum.h"

struct particle
{
	uint cell_and_existence;
	uint3 position;

	__device__ __host__ particle() : position(make_uint3(0u)), cell_and_existence(0u) {}

	__device__ __host__ bool exists() const { return cell_and_existence >> 31u; }
	__device__ __host__ uint in_cell_index() const { return cell_and_existence & 2147483647u; }
	__device__ __host__ void set_cell_index(const uint idx) { cell_and_existence = (exists() << 31u) | (idx & 2147483647u); }
	__device__ __host__ void set_existence(const bool exists) { cell_and_existence = (exists << 31u) | (cell_and_existence & 2147483647u); }

	__device__ __host__ uint morton_index() const {
		uint result = 0u;
		for (uint i = 0u; i < grid_dimension_pow; i++)
		{
			result |= ((position.x >> (i + 32u - grid_dimension_pow)) & 1u) << (3u * i) |
				 ((position.y >> (i + 32u - grid_dimension_pow)) & 1u) << (3u * i + 1u) |
				 ((position.z >> (i + 32u - grid_dimension_pow)) & 1u) << (3u * i + 2u);
		}
		return result;
	}
	__device__ __host__ float3 true_pos() const
	{
		return make_float3(position) * (domain_size_km / 4294967295.f);
	}
	__device__ __host__ void set_true_pos(float3 pos)
	{
		pos /= domain_size_km;
		if (wrap_around)
			pos -= floorf(pos);
		if (fminf(global_min(pos), 0.999999940395355224609375f - global_max(pos)) < 0.f)
			cell_and_existence = 0u;
		else
			position = make_uint3(pos * 4294967296.f);
	}
};
static_assert(sizeof(particle) == 16, "Wrong size!");

struct particle_kinematics
{
	float mass_Tg;
	float radius_km;
	float3 velocity_kms;
	float3 acceleration_ms2;
	__device__ __host__ void multiply_radius(float change_factor = 1.f, float min_rad = 100.f, float max_rad = 1000.f)
	{
		radius_km = fminf(fmaxf(radius_km * change_factor, min_rad), max_rad);
	}
	__device__ __host__ void change_mass(particle* particle, float change_Tg = 0.f)
	{
		mass_Tg += change_Tg;
		if (mass_Tg <= 0.f)
			particle->cell_and_existence = 0u;
	}
	__device__ __host__ particle_kinematics() : velocity_kms(make_float3(0.f)), acceleration_ms2(make_float3(0.f)), mass_Tg(1E-40f), radius_km(0.f) {}
};
static_assert(sizeof(particle_kinematics) == 32, "Wrong size!");

template <class T>
struct particle_data_buffer
{
	smart_gpu_buffer<T> buffer;
	smart_gpu_buffer<T> temp;
	particle_data_buffer(size_t dedicated_len) : buffer(dedicated_len), temp(dedicated_len) {}
	void swap_pointers() { buffer.swap_pointers(temp); }
	void destroy() { buffer.destroy(); temp.destroy(); }
};

//////////////////////////////////////
//// Spatial Sort with Count Sort ////
//////////////////////////////////////

template <class T>
__global__ void __set_empty(T* buffer, uint length)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= length) { return; }
	buffer[idx] = T();
}

__global__ void __locate_in_cells(uint* cell_counts, particle* particles, uint number_particles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= number_particles) { return; }
	if (particles[idx].exists())
		particles[idx].set_cell_index(atomicAdd(&cell_counts[particles[idx].morton_index()], 1u));
}

template <class T>
__global__ void __copy_spatial_counting_sort_data(const uint* cell_pos, const particle* target, const T* old_buffer, T* new_buffer, uint number_particles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= number_particles) { return; }

	particle part = target[idx];
	if (part.exists())
		new_buffer[__read_start_idx(cell_pos, part.morton_index()) + part.in_cell_index()] = old_buffer[idx];
}

__global__ void __copy_spatial_counting_sort(const uint* cell_pos, const particle* old_buffer, particle* new_buffer, uint number_particles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= number_particles) { return; }

	particle part = old_buffer[idx];
	if (part.exists())
		new_buffer[__read_start_idx(cell_pos, part.morton_index()) + part.in_cell_index()] = part;
}

template <class T>
void counting_sort_data_transfer(const smart_gpu_buffer<uint>& cell_bounds, const smart_gpu_buffer<particle>& targets, particle_data_buffer<T>& buffer)
{
	dim3 threads(targets.dedicated_len > 512u ? 512u : targets.dedicated_len);
	dim3 blocks((uint)ceilf(targets.dedicated_len / (float)threads.x));
	__copy_spatial_counting_sort_data<T><<<blocks, threads>>>(cell_bounds.gpu_buffer_ptr, targets.gpu_buffer_ptr, buffer.buffer.gpu_buffer_ptr, buffer.temp.gpu_buffer_ptr, targets.dedicated_len);
	buffer.swap_pointers();
}

//////////////////////////////////
////   Geometry Initializers  ////
//////////////////////////////////

__global__ void __init_sphere(particle* particles, const uint particle_count, const uint offset, const uint layers, const float3 center, const float particle_spacing, const float padding)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_count) { return; }

	uint layer = clamp(cbrtf(idx / (float)particle_count) * layers, .0001f, layers - .0001f);

	float lower_bounds = layer / ((float)layers);
	lower_bounds *= lower_bounds * lower_bounds * particle_count;
	float upper_bounds = clamp((layer + 1.f) / ((float)layers), 0.f, 1.f);
	upper_bounds *= upper_bounds * upper_bounds * particle_count;

	float z = 1.f - ((idx - lower_bounds) * 2.f / (upper_bounds - lower_bounds));
	float t = 3.88322207745f * (idx - lower_bounds), r = sqrtf(1.f - z * z);
	float lc = cosf(3.88322207745f * layer), ls = sinf(3.88322207745f * layer), ts = sinf(t);
	particle to_set = particle();

	to_set.set_existence(true);
	to_set.set_true_pos(center + make_float3(cosf(t) * r, (ts * r) * lc + z * ls, z * lc - (ts * r) * ls) * (layer * .85f + padding) * particle_spacing);
	particles[idx + offset] = to_set;
}

__global__ void __init_cuboid(particle* particles, const uint particle_count, const uint offset, const uint3 layers, const float3 center, const float particle_spacing)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_count) { return; }

	particle to_set = particle();
	uint3 positions_in_shape = make_uint3(idx % layers.x, (idx / layers.x) % layers.y, idx / (layers.x * layers.y));
	float3 positions = make_float3(positions_in_shape);
	positions.x += ((positions_in_shape.y % 2u) + (positions_in_shape.z % 2u)) * .5f - .5f;
	positions.y = (positions.y + (positions_in_shape.z % 2u) * .5f - .5f) * 0.86602540378f; positions.z *= 0.75f;
	positions.x += positions_in_shape.x * .3333333333f / (float)layers.x - .1666666666f;

	to_set.set_existence(true);
	to_set.set_true_pos(center + (positions - make_float3(layers - 1u) * make_float3(.5f, 0.43301270189f, .375f)) * particle_spacing);
	particles[idx + offset] = to_set;
}

////////////////////////////////////
////		   Dynamics			////
////////////////////////////////////

__global__ void __init_kinematics(const particle* particles, particle_kinematics* kinematics, const uint particle_count, const uint offset, const float3 velocity,
	const float3 center, const float3 angular_velocity, const float particle_mass, const float particle_radius)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_count) { return; }

	float3 position_from_center = particles[idx + offset].true_pos() - center;
	float3 tangential_vel = cross(angular_velocity, position_from_center);

	particle_kinematics kinematics_to_set = particle_kinematics();
	float dst_center = length(position_from_center);
	kinematics_to_set.mass_Tg = particle_mass;
	kinematics_to_set.radius_km = particle_radius;
	kinematics_to_set.velocity_kms = velocity + tangential_vel;
	kinematics[idx + offset] = kinematics_to_set;
}

__global__ void __apply_kinematics(particle* particles, particle_kinematics* kinematics, const float timestep_s, const uint particle_capacity, const float3 subtract_offset)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	if (!particles[idx].exists()) { return; }
	float3 new_V = kinematics[idx].velocity_kms + kinematics[idx].acceleration_ms2 * timestep_s * .001f;
	particles[idx].set_true_pos(particles[idx].true_pos() + new_V * timestep_s - subtract_offset);
	kinematics[idx].velocity_kms = new_V;
	kinematics[idx].acceleration_ms2 = make_float3(0.f);
}

//////////////////////////////////
////	  Main Structures	  ////
//////////////////////////////////

struct kinematic_simulation
{
	size_t particle_capacity;
	particle_data_buffer<particle_kinematics> kinematic_data;
	particle_data_buffer<particle> particles;
	smart_gpu_buffer<uint> cell_bounds;

	void set_cuboid(const uint particle_count, const uint write_offset, const float total_mass_Tg, const float3 dimensions, const float3 center_km = make_float3(domain_size_km * .5f), const float3 velocity_kms = make_float3(0.f), const float3 angular_vel_rads = make_float3(0.f))
	{
		uint threads = min(particle_count, 512);
		uint blocks = ceilf(particle_count / (float)threads);
		float average_radius = cbrtf(dimensions.x * dimensions.y * dimensions.z / particle_count);
		uint3 layers = make_uint3(roundf(dimensions * make_float3(0.86602540378f, 1.f, 1.15470053838f) / average_radius));
		layers.z = (uint)roundf(particle_count / ((float)layers.x * layers.y));

		__init_cuboid<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, particle_count, write_offset, layers, center_km, average_radius);
		__init_kinematics<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particle_count, write_offset, velocity_kms,
			center_km, angular_vel_rads, total_mass_Tg / particle_count, average_radius * 1.15470053838f); cuda_sync();
	}
	void set_sphere(const uint particle_count, const uint write_offset, const float total_mass_Tg, const float total_radius_km, const float3 center_km = make_float3(domain_size_km * .5f), const float3 velocity_kms = make_float3(0.f), const float3 angular_vel_rads = make_float3(0.f))
	{
		uint threads = min(particle_count, 512);
		uint blocks = ceilf(particle_count / (float)threads);
		uint layers = ceilf(cbrtf(particle_count) * .85f);

		__init_sphere<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, particle_count, write_offset, layers, center_km, total_radius_km / (layers * .85f - 0.2f), (particle_count > 1u) * .15f);
		__init_kinematics<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particle_count, write_offset, velocity_kms,
			center_km, angular_vel_rads, total_mass_Tg / particle_count, total_radius_km / cbrtf(particle_count)); cuda_sync();
	}
	
	/// <summary>
	/// Copy data associated with particles in sort. Override for required data.
	/// </summary>
	/// <param name="cell_bounds">: Cell boundaries.</param>
	/// <param name="targets">: Target particles.</param>
	virtual void counting_sort_transfers(const smart_gpu_buffer<uint>& cell_bounds, const smart_gpu_buffer<particle>& targets) { }
	kinematic_simulation(size_t allocation_particles) : particles(allocation_particles), particle_capacity(allocation_particles), cell_bounds(grid_cell_count), kinematic_data(allocation_particles) {
		dim3 threads(particles.buffer.dedicated_len > 512u ? 512u : particles.buffer.dedicated_len);
		dim3 blocks((uint)ceilf(particles.buffer.dedicated_len / (float)threads.x));

		__set_empty<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, particles.buffer.dedicated_len);
	}
	void apply_kinematics(float timestep_s = 1.f)
	{
		uint threads = particle_capacity < 512u ? particle_capacity : 512u;
		uint blocks = ceilf(particle_capacity / (float)threads);

		__apply_kinematics<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, timestep_s, particle_capacity, make_float3(0.f));
		cuda_sync();
	}
	void sort_spatially()
	{
		dim3 threads(particle_capacity > 512u ? 512u : particle_capacity);
		dim3 blocks((uint)ceilf(particle_capacity / (float)threads.x));

		__set_empty<<<blocks, threads>>>(particles.temp.gpu_buffer_ptr, particle_capacity);
		__set_empty<<<((uint)ceilf(cell_bounds.dedicated_len / 512.f)), 512u>>>(cell_bounds.gpu_buffer_ptr, cell_bounds.dedicated_len);
	
		__locate_in_cells<<<blocks, threads>>>(cell_bounds.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particle_capacity);
		apply_incl_cum_sum<uint>(cell_bounds); counting_sort_data_transfer(cell_bounds, particles.buffer, kinematic_data);
		counting_sort_transfers(cell_bounds, particles.buffer); cuda_sync();

		__copy_spatial_counting_sort<<<blocks, threads>>>(cell_bounds.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particles.temp.gpu_buffer_ptr, particle_capacity);
		cuda_sync(); particles.swap_pointers();
	}
};

#include <vector>
struct initial_kinematic_object
{
	enum geometry
	{
		GEOM_SPHERE,
		GEOM_CUBOID
	};

	float3 angular_velocity_rads;
	float3 velocity_kms;
	float3 center_pos_km;
	float total_mass_Tg;

	geometry geometry_type;
	std::vector<float> dimensions;

	initial_kinematic_object() {}

	float particle_assignment_weightage() const {
		float temp_var;
		switch (geometry_type)
		{
		case initial_kinematic_object::GEOM_SPHERE:
			temp_var = dimensions[0] / domain_size_km;
			temp_var *= temp_var * temp_var * 4.18879020479f;
			break;
		case initial_kinematic_object::GEOM_CUBOID:
			temp_var = dimensions[0] / domain_size_km;
			temp_var *= dimensions[1] / domain_size_km;
			temp_var *= dimensions[2] / domain_size_km;
			break;
		}
		return expf(logf(temp_var) * .75f + logf(total_mass_Tg) * .25f);
	}
	initial_kinematic_object(geometry geometry_type, std::vector<float> dimensions, float total_mass_Tg, 
		float3 center_pos_km = make_float3(domain_size_km * .5f), float3 velocity_kms = make_float3(0.f), float3 angular_velocity_rads = make_float3(0.f)) : 
		dimensions(dimensions), geometry_type(geometry_type), total_mass_Tg(total_mass_Tg), center_pos_km(center_pos_km), velocity_kms(velocity_kms), angular_velocity_rads(angular_velocity_rads)
	{
		switch (geometry_type)
		{
			case GEOM_SPHERE:
				if (dimensions.size() != 1u)
					throw std::invalid_argument("Wrong number of dimensions! Should only take radius!");
				break;
			case GEOM_CUBOID:
				if (dimensions.size() != 3u)
					throw std::invalid_argument("Wrong number of dimensions! Should only take axis dimensions!");
				break;
		}
	}
};
std::vector<uint> find_apportionment_hamilton(const uint max_particles, const std::vector<initial_kinematic_object>& objects, float3* avg_vel)
{
	const uint object_count = objects.size();
	std::vector<uint> apportionment(object_count);

	float total_volume = 0.f, total_mass = 0.f;
	for (uint i = 0u; i < object_count; i++)
		total_volume += objects[i].particle_assignment_weightage();

	uint used_count = 0u;
	if (avg_vel != nullptr) { *avg_vel = make_float3(0.f); }
	std::vector<float> truncation_error(object_count);
	for (uint i = 0u; i < object_count; i++)
	{
		float target_particles = (float)max_particles * objects[i].particle_assignment_weightage() / total_volume;
		apportionment[i] = (uint)fmaxf(1.5f, target_particles);
		truncation_error[i] = target_particles - apportionment[i];
		used_count += apportionment[i];

		if (avg_vel == nullptr) { continue; }
		total_mass += objects[i].total_mass_Tg * 1E-12f;
		*avg_vel += objects[i].velocity_kms * (objects[i].total_mass_Tg * 1E-12f);
	}
	if (avg_vel != nullptr) { *avg_vel /= fmaxf(total_mass, 1E-20f); }

	while (used_count < max_particles)
	{
		uint tr_idx = 0u;
		float max_trunc_error = truncation_error[0];
		for (uint i = 1u; i < object_count; i++)
			if (truncation_error[i] > max_trunc_error)
			{
				max_trunc_error = truncation_error[i];
				tr_idx = i;
			}

		apportionment[tr_idx]++;
		truncation_error[tr_idx]--;
		used_count++;
	}
	while (used_count > max_particles)
	{
		uint max_apportionment_idx = 0u;
		float max_apportionment = -INFINITY;
		for (uint i = 0u; i < object_count; i++)
		{
			if (apportionment[i] <= 1)
				continue;
			float ratio = (float)apportionment[i] * total_volume /
				((float)max_particles * objects[i].particle_assignment_weightage());
			if (ratio > max_apportionment)
			{
				max_apportionment = ratio;
				max_apportionment_idx = i;
			}
		}

		apportionment[max_apportionment_idx]--;
		used_count--;
	}
	return apportionment;
}
std::vector<uint> initialize_kinematic_objects(kinematic_simulation& simulation, const std::vector<initial_kinematic_object>& objects, bool center_of_mass_frame = true)
{
	uint offsets = 0u; const uint object_count = objects.size();
	if (object_count == 0u) { return std::vector<uint>(0u); } float3 average_vel;
	std::vector<uint> particle_counts = find_apportionment_hamilton(simulation.particle_capacity, objects, &average_vel);
	if (!center_of_mass_frame) { average_vel = make_float3(0.f); }

	for (uint i = 0u; i < object_count; i++)
	{
		const initial_kinematic_object& obj = objects[i];
		switch (obj.geometry_type)
		{
			case initial_kinematic_object::GEOM_SPHERE:
				simulation.set_sphere(particle_counts[i], offsets, obj.total_mass_Tg, obj.dimensions[0], obj.center_pos_km, obj.velocity_kms - average_vel, obj.angular_velocity_rads);
				writeline("Creating sphere with particle count " + std::to_string(particle_counts[i]));
				break;
			case initial_kinematic_object::GEOM_CUBOID:
				simulation.set_cuboid(particle_counts[i], offsets, obj.total_mass_Tg, make_float3(obj.dimensions[0], obj.dimensions[1], obj.dimensions[2]), obj.center_pos_km, obj.velocity_kms - average_vel, obj.angular_velocity_rads);
				writeline("Creating cuboid with particle count " + std::to_string(particle_counts[i]));
				break;
		}
		offsets += particle_counts[i];
	}
	writeline("");
	return particle_counts;
}

#endif