#ifndef GRAVITATION_H
#define GRAVITATION_H
#include "spatial_grid.h"

// Tunable
__device__ constexpr bool compute_gravity_empty_cell = false;
__device__ constexpr float barnes_hut_criterion = 0.35f;
__device__ constexpr float G_in_Tg_km_units = 6.6743015E-8f;

// Derived
__device__ constexpr uint block_size_barnes_hut = 64u >> (grid_dimension_pow - minimum_depth > 4); // constrained by shared memory
__device__ constexpr uint min_stack_size_required = 2u + 7u * (grid_dimension_pow - minimum_depth);

struct particle_kinematics
{
	float mass_Tg;
	float radius_km;
	float3 velocity_kms;
	float3 acceleration_ms2;
	__device__ __host__ particle_kinematics() : velocity_kms(make_float3(0.f)), acceleration_ms2(make_float3(0.f)), mass_Tg(1E-40f), radius_km(0.f) {}
};
struct grid_cell_ensemble
{
	float standard_radius_km;
	float total_mass_Tg;
	float3 deviatoric_pos_km;
	float3 average_acceleration_ms2;
	__device__ __host__ grid_cell_ensemble() : deviatoric_pos_km(make_float3(0.f)), total_mass_Tg(1E-40f), standard_radius_km(0.f) {}

	__device__ __host__ float average_density_kgm3() const { return total_mass_Tg / fmaxf(4.18879020479f * standard_radius_km * standard_radius_km * standard_radius_km, 1E-5f); }
};

__device__ __host__ float grav_force_diffuse_unfactored(float separation, float standard_deviation)
{
	standard_deviation *= 1.41421356237f;
	if (separation < .25f * standard_deviation)
		return -0.75225277806f / (standard_deviation * standard_deviation * standard_deviation);
	if (separation > 3.f * standard_deviation)
		return -1.f / (separation * separation * separation);
	return 1.1283791671f * expf(-separation * separation / (standard_deviation * standard_deviation)) / (separation * separation * standard_deviation) - erf_lossy(separation / standard_deviation) / (separation * separation * separation);
}
__device__ __host__ constexpr uint __start_index(uint depth)
{
	return (mask_morton_x & ((~0u) >> (32u - depth * 3u))) - (mask_morton_x & ((~0u) >> (32u - minimum_depth * 3u)));
}

__global__ void __average_ensemble(const particle_kinematics* kinematics, const particle* positions, const uint* cell_pos, grid_cell_ensemble* grid_cells)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= grid_cell_count) { return; }
	const uint start_pos = __read_start_idx(cell_pos, idx), end_pos = __read_start_idx(cell_pos, idx + 1);
	const float3 this_cell_pos = __cell_pos_from_index(idx, grid_dimension_pow);

	grid_cell_ensemble this_ensemble = grid_cell_ensemble();
	float3 variances = make_float3(0.f);

	for (uint i = start_pos; i < end_pos; i++)
	{
		float mass = kinematics[i].mass_Tg;
		float radius = kinematics[i].radius_km;
		float3 rel_pos = positions[i].true_pos() - this_cell_pos;

		this_ensemble.total_mass_Tg += mass;
		this_ensemble.deviatoric_pos_km += rel_pos * mass;
		variances += (rel_pos * rel_pos + radius * radius * .3333333333f) * mass;
	}

	this_ensemble.deviatoric_pos_km /= this_ensemble.total_mass_Tg;
	variances = (variances / this_ensemble.total_mass_Tg) - this_ensemble.deviatoric_pos_km * this_ensemble.deviatoric_pos_km;
	this_ensemble.standard_radius_km = sqrtf(variances.x + variances.y + variances.z);
	grid_cells[idx + __start_index(grid_dimension_pow)] = this_ensemble;
}
__global__ void __mipmap_1_layer(grid_cell_ensemble* grid_cells, const uint target_depth)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= 1u << (3u * target_depth)) { return; }

	grid_cell_ensemble this_ensemble = grid_cell_ensemble();
	const float3 this_cell_pos = __cell_pos_from_index(idx, target_depth);
	float3 variances = make_float3(0.f);

	for (uint i = idx * 8; i < (idx + 1) * 8; i++)
	{
		grid_cell_ensemble target = grid_cells[i + __start_index(target_depth + 1)];
		float3 rel_pos = target.deviatoric_pos_km + __cell_pos_from_index(i, target_depth + 1) - this_cell_pos;

		this_ensemble.total_mass_Tg += target.total_mass_Tg;
		this_ensemble.deviatoric_pos_km += rel_pos * target.total_mass_Tg;
		variances += (rel_pos * rel_pos + target.standard_radius_km * target.standard_radius_km * .333333333f) * target.total_mass_Tg;
	}

	this_ensemble.deviatoric_pos_km /= this_ensemble.total_mass_Tg;
	variances = (variances / this_ensemble.total_mass_Tg) - this_ensemble.deviatoric_pos_km * this_ensemble.deviatoric_pos_km;
	this_ensemble.standard_radius_km = sqrtf(variances.x + variances.y + variances.z);
	grid_cells[idx + __start_index(target_depth)] = this_ensemble;
}

struct octree_indexer
{
	uint data;
	__device__ __host__ octree_indexer(const bool invalid = true) : data(invalid * (~0u)) {}
	__device__ __host__ octree_indexer(const uint depth, const uint morton_index) : data((depth & 7u) << 29u | (morton_index & ((~0u) >> 3u))) {}
	__device__ __host__ uint depth() const { return data >> 29u; }
	__device__ __host__ uint morton_index() const { return data & ((~0u) >> 3u); }
	__device__ __host__ bool valid() const { return ~data; }
};
__device__ __host__ uint stack_content(octree_indexer* ptrs)
{
	return ptrs[0].data;
}
__device__ __host__ void add_to_stack(octree_indexer* ptrs, octree_indexer data)
{
	ptrs[ptrs[0].data += 1u] = data;
}
__device__ __host__ octree_indexer pop_stack(octree_indexer* ptrs)
{
	uint loc = ptrs[0].data;
	octree_indexer result = loc > 0u ? ptrs[loc] : octree_indexer();
	ptrs[0].data -= loc > 0u; return result;
}
__device__ __host__ octree_indexer compute_gravity_check(const float3 this_pos, const grid_cell_ensemble* octree, octree_indexer* stack, float3& acceleration_ms2)
{
	const octree_indexer s = pop_stack(stack);
	if (!s.valid()) { return s; }

	const uint other_m = s.morton_index(), other_d = s.depth();
	const grid_cell_ensemble other_c = octree[other_m + __start_index(other_d)];
	float3 separation = this_pos - (other_c.deviatoric_pos_km + __cell_pos_from_index(other_m, other_d));
	if ((other_c.standard_radius_km * other_c.standard_radius_km / dot(separation, separation) > barnes_hut_criterion * barnes_hut_criterion) && (other_d < grid_dimension_pow))
		return s;

	acceleration_ms2 += separation * (other_c.total_mass_Tg * G_in_Tg_km_units * grav_force_diffuse_unfactored(length(separation) + 1E-10f, other_c.standard_radius_km + 1E-10f));
	return octree_indexer();
}

__global__ void __compute_barnes_hut(grid_cell_ensemble* octree, const uint* cell_bounds)
{
	uint idx = threadIdx.x + block_size_barnes_hut * blockIdx.x; // block_size_barnes_hut = 64u
	if (idx >= grid_cell_count) { return; }
	float3 acceleration_ms2 = make_float3(0.f);
	if (compute_gravity_empty_cell || __count_particles(cell_bounds, idx, grid_dimension_pow) > 0) {
		__shared__ octree_indexer stacks[min_stack_size_required * block_size_barnes_hut]; // min_stack_size_required * block_size_barnes_hut = 1920u
		octree_indexer* this_stack = stacks + threadIdx.x * min_stack_size_required;
		float3 this_position_km = octree[idx + __start_index(grid_dimension_pow)].deviatoric_pos_km + __cell_pos_from_index(idx, grid_dimension_pow);
		for (uint i = 0; i < 1u << (minimum_depth * 3u); i++)
		{
			this_stack[0].data = 0u;
			add_to_stack(this_stack, octree_indexer(minimum_depth, i));

			while (stack_content(this_stack) > 0u)
			{
				octree_indexer curr_check = compute_gravity_check(this_position_km, octree, this_stack, acceleration_ms2);
				if (curr_check.valid())
					for (uint i = 0u; i < 8u; i++)
						add_to_stack(this_stack, octree_indexer(curr_check.depth() + 1u, (curr_check.morton_index() << 3u) + i));
			}
		}
	}
	octree[idx + __start_index(grid_dimension_pow)].average_acceleration_ms2 = acceleration_ms2;
}
__global__ void __init_kinematics(const particle* particles, particle_kinematics* kinematics, const uint particle_count, const uint offset, const float3 velocity,
								const float3 center, const float3 angular_velocity, const float particle_mass, const float particle_radius)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_count) { return; }

	float3 position_from_center = particles[idx + offset].true_pos() - center;
	float3 tangential_vel = make_float3(position_from_center.z * angular_velocity.y - position_from_center.y * angular_velocity.z,
										-position_from_center.z * angular_velocity.x + position_from_center.x * angular_velocity.z,
										position_from_center.y * angular_velocity.x - position_from_center.x * angular_velocity.y);

	particle_kinematics kinematics_to_set = particle_kinematics();
	kinematics_to_set.mass_Tg = particle_mass;
	kinematics_to_set.radius_km = particle_radius;
	kinematics_to_set.velocity_kms = velocity + tangential_vel;
	kinematics[idx + offset] = kinematics_to_set;
}
__global__ void __apply_gravitation(const grid_cell_ensemble* octree, const uint* cell_bounds, const particle* particles, particle_kinematics* kinematics, const uint particle_capacity)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	if (!particles[idx].exists()) { return; }
	float3 this_pos = particles[idx].true_pos();
	uint morton_index = particles[idx].morton_index();
	float3 acceleration_ms2 = octree[morton_index + __start_index(grid_dimension_pow)].average_acceleration_ms2;

	for (uint i = __read_start_idx(cell_bounds, morton_index), j = __read_start_idx(cell_bounds, morton_index + 1u); i < j; i++)
	{
		if (i == idx) { continue; }
		float3 separation = this_pos - particles[i].true_pos();
		acceleration_ms2 += separation * (kinematics[i].mass_Tg * G_in_Tg_km_units * grav_force_diffuse_unfactored(length(separation) + 1E-10f, kinematics[i].radius_km + 1E-10f));
	}

	kinematics[idx].acceleration_ms2 = acceleration_ms2;
}
__global__ void __apply_kinematics(particle* particles, particle_kinematics* kinematics, const float timestep_s, const uint particle_capacity, const float3 subtract_offset)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	if (!particles[idx].exists()) { return; }
	float3 new_V = kinematics[idx].velocity_kms + kinematics[idx].acceleration_ms2 * timestep_s * .001f;
	particles[idx].set_true_pos(particles[idx].true_pos() + new_V * timestep_s - subtract_offset);
	kinematics[idx].velocity_kms = new_V;
}

struct gravitational_simulation : spatial_grid
{
	particle_data_buffer<particle_kinematics> kinematic_data;
	smart_gpu_buffer<grid_cell_ensemble> octree;
private:
	smart_cpu_buffer<grid_cell_ensemble> coarsest;
public:

	void set_massive_cuboid(const uint particle_count, const uint write_offset, const float total_mass_Tg, const float3 dimensions, const float3 center_km = make_float3(domain_size_km * .5f), const float3 velocity_kms = make_float3(0.f), const float3 angular_vel_rads = make_float3(0.f))
	{
		uint threads = min(particle_count, 512);
		uint blocks = ceilf(particle_count / (float)threads);
		float average_radius = cbrtf(dimensions.x * dimensions.y * dimensions.z / particle_count);

		set_point_cuboid(particle_count, write_offset, dimensions, center_km);
		__init_kinematics<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particle_count, write_offset, velocity_kms,
			center_km, angular_vel_rads, total_mass_Tg / particle_count, average_radius); cuda_sync();
	}
	void set_massive_sphere(const uint particle_count, const uint write_offset, const float total_mass_Tg, const float total_radius_km, const float3 center_km = make_float3(domain_size_km * .5f), const float3 velocity_kms = make_float3(0.f), const float3 angular_vel_rads = make_float3(0.f))
	{
		uint threads = min(particle_count, 512);
		uint blocks = ceilf(particle_count / (float)threads);

		set_point_sphere(particle_count, write_offset, total_radius_km, center_km);
		__init_kinematics<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particle_count, write_offset, velocity_kms,
			center_km, angular_vel_rads, total_mass_Tg / particle_count, total_radius_km / cbrtf(particle_count)); cuda_sync();
	}
	virtual void counting_sort_transfers(const smart_gpu_buffer<uint>& cell_bounds, const smart_gpu_buffer<particle>& targets) override
	{
		counting_sort_data_transfer(cell_bounds, targets, kinematic_data);
	}
	gravitational_simulation(size_t allocation_particles) : spatial_grid(allocation_particles), octree(__start_index(grid_dimension_pow + 1)), kinematic_data(allocation_particles), coarsest(1u << (3u * minimum_depth)) { }
	void generate_gravitational_data()
	{
		__average_ensemble<<<((uint)ceilf(cell_bounds.dedicated_len / 512.f)), 512u>>>(kinematic_data.buffer.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr, octree.gpu_buffer_ptr);
		for (int i = grid_dimension_pow - 1; i >= minimum_depth; i--)
		{
			dim3 threads(min(512u, 1u << (3u * i)));
			__mipmap_1_layer<<<((uint)ceilf((1u << (3u * i)) / (float)threads.x)), threads>>>(octree.gpu_buffer_ptr, i);
		}
		__compute_barnes_hut<<<((uint)ceilf(grid_cell_count / (float)block_size_barnes_hut)), block_size_barnes_hut>>>(octree.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr); cuda_sync();
	}
	void apply_kinematics(float timestep_s = 1.f, bool recenter = true)
	{
		uint threads = particle_capacity < 512u ? particle_capacity : 512u;
		uint blocks = ceilf(particle_capacity / (float)threads);

		float3 avg_pos = make_float3(0.f);
		if (recenter)
		{
			cudaMemcpy(coarsest.cpu_buffer_ptr, octree.gpu_buffer_ptr, (1u << (3u * minimum_depth)) * sizeof(grid_cell_ensemble), cudaMemcpyDeviceToHost);

			float total_mass = 1E-30f;
			for (uint i = 0u; i < 1u << (3u * minimum_depth); i++)
			{
				grid_cell_ensemble target = coarsest.cpu_buffer_ptr[i];
				avg_pos += (target.deviatoric_pos_km + __cell_pos_from_index(i, minimum_depth)) * target.total_mass_Tg;
				total_mass += target.total_mass_Tg;
			}
			avg_pos = (avg_pos / total_mass) - domain_size_km * .5f;
		}

		__apply_kinematics<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, timestep_s, particle_capacity, avg_pos);
	}
	void apply_gravitation()
	{
		uint threads = particle_capacity < 512u ? particle_capacity : 512u;
		uint blocks = ceilf(particle_capacity / (float)threads);

		__apply_gravitation<<<blocks, threads>>>(octree.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particle_capacity);
	}

};

#endif