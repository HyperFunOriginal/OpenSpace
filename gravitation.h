#ifndef GRAVITATION_H
#define GRAVITATION_H
#include "spatial_grid.h"

struct particle_kinematics
{
	float mass_Tg;
	float radius_km;
	float3 velocity_kms;
	float3 acceleration_kms;
	__device__ __host__ particle_kinematics() : velocity_kms(make_float3(0.f)), acceleration_kms(make_float3(0.f)), mass_Tg(1E-40f), radius_km(0.f) {}
};
struct grid_cell_ensemble
{
	float standard_radius_km;
	float total_mass_Tg;
	float3 deviatoric_pos_km;
	float3 average_acceleration_ms2;
	__device__ __host__ grid_cell_ensemble() : deviatoric_pos_km(make_float3(0.f)), total_mass_Tg(1E-40f), standard_radius_km(0.f) {}
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
__device__ __host__ constexpr uint start_index(uint depth)
{
	return (mask_morton_x & ((~0u) >> (32u - depth * 3u))) - (mask_morton_x & ((~0u) >> (32u - minimum_depth * 3u)));
}

__global__ void __average_ensemble(const particle_kinematics* kinematics, const particle* positions, const uint* cell_pos, grid_cell_ensemble* grid_cells)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= grid_side_length * grid_side_length * grid_side_length) { return; }
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
	grid_cells[idx + start_index(grid_dimension_pow)] = this_ensemble;
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
		grid_cell_ensemble target = grid_cells[i + start_index(target_depth + 1)];
		float3 rel_pos = target.deviatoric_pos_km + __cell_pos_from_index(i, target_depth + 1) - this_cell_pos;

		this_ensemble.total_mass_Tg += target.total_mass_Tg;
		this_ensemble.deviatoric_pos_km += rel_pos * target.total_mass_Tg;
		variances += (rel_pos * rel_pos + target.standard_radius_km * target.standard_radius_km * .333333333f) * target.total_mass_Tg;
	}

	this_ensemble.deviatoric_pos_km /= this_ensemble.total_mass_Tg;
	variances = (variances / this_ensemble.total_mass_Tg) - this_ensemble.deviatoric_pos_km * this_ensemble.deviatoric_pos_km;
	this_ensemble.standard_radius_km = sqrtf(variances.x + variances.y + variances.z);
	grid_cells[idx + start_index(target_depth)] = this_ensemble;
}

__device__ constexpr bool compute_gravity_empty_cell = true;
__device__ constexpr uint block_size_barnes_hut = 64u >> (grid_dimension_pow - minimum_depth > 4);
__device__ constexpr float barnes_hut_criterion = 0.5f;
__device__ constexpr float G_in_Tg_km_units = 6.6743015E-8f;
__device__ constexpr uint min_stack_size_required = 2u + 7u * (grid_dimension_pow - minimum_depth);

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
	const grid_cell_ensemble other_c = octree[other_m + start_index(other_d)];
	float3 separation = this_pos - (other_c.deviatoric_pos_km + __cell_pos_from_index(other_m, other_d));
	if ((other_c.standard_radius_km * other_c.standard_radius_km / dot(separation, separation) > barnes_hut_criterion * barnes_hut_criterion) && (other_d < grid_dimension_pow))
		return s;

	acceleration_ms2 += separation * (other_c.total_mass_Tg * G_in_Tg_km_units * grav_force_diffuse_unfactored(length(separation) + 1E-10f, other_c.standard_radius_km + 1E-10f));
	return octree_indexer();
}

__global__ void __compute_barnes_hut(grid_cell_ensemble* octree, const uint* cell_bounds)
{
	uint idx = threadIdx.x + block_size_barnes_hut * blockIdx.x; // block_size_barnes_hut = 64u
	if (idx >= grid_side_length * grid_side_length * grid_side_length) { return; }
	float3 acceleration_ms2 = make_float3(0.f);
	if (compute_gravity_empty_cell || __count_particles(cell_bounds, idx, grid_dimension_pow) > 0) {
		__shared__ octree_indexer stacks[min_stack_size_required * block_size_barnes_hut]; // min_stack_size_required * block_size_barnes_hut = 1920u
		octree_indexer* this_stack = stacks + threadIdx.x * min_stack_size_required;
		float3 this_position_km = octree[idx + start_index(grid_dimension_pow)].deviatoric_pos_km + __cell_pos_from_index(idx, grid_dimension_pow);
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
	octree[idx + start_index(grid_dimension_pow)].average_acceleration_ms2 = acceleration_ms2;
}
__global__ void __init_sphere(particle* particles, particle_kinematics* kinematics, const uint particle_count, const uint offset, 
	const uint layers, const float3 center, const float3 velocity, const float particle_mass, const float particle_radius)
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
	particle to_set = particle();

	to_set.set_existence(true);
	to_set.set_true_pos(center + make_float3(cosf(t) * r, sinf(t) * r, z) * (layer * .7f + .5f) * particle_radius);
	particles[idx + offset] = to_set;

	particle_kinematics kinematics_to_set = particle_kinematics();
	kinematics_to_set.mass_Tg = particle_mass;
	kinematics_to_set.radius_km = particle_radius;
	kinematics_to_set.velocity_kms = velocity;
	kinematics[idx + offset] = kinematics_to_set;
}

struct gravitational_simulation : spatial_grid
{
	particle_data_buffer<particle_kinematics> kinematic_data;
	smart_gpu_cpu_buffer<grid_cell_ensemble> octree;

	virtual void counting_sort_transfers(const smart_gpu_buffer<uint>& cell_bounds, const smart_gpu_buffer<particle>& targets) override
	{
		counting_sort_data_transfer(cell_bounds, targets, kinematic_data);
	}
	virtual void set_sphere(const uint particle_count, const uint write_offset, const float total_mass_Tg, const float total_radius_km, const float3 center_km = make_float3(domain_size_km * .5f), const float3 velocity_kms = make_float3(0.f))
	{
		uint threads = min(particle_count, 512);
		uint blocks = ceilf(particle_count / (float)threads);
		uint layers = ceilf(cbrtf(particle_count) * .55f);

		__init_sphere<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particle_count, write_offset, layers,
			center_km, velocity_kms, total_mass_Tg / particle_count, total_radius_km / (layers * .9f - 0.2f));
		cuda_sync();
	}
	gravitational_simulation(size_t allocation_particles) : spatial_grid(allocation_particles), octree(start_index(grid_dimension_pow + 1)), kinematic_data(allocation_particles) { }
	void generate_gravitational_data()
	{
		__average_ensemble<<<((uint)ceilf(cell_bounds.dedicated_len / 512.f)), 512u>>>(kinematic_data.buffer.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr, octree.gpu_buffer_ptr);
		for (int i = grid_dimension_pow - 1; i >= minimum_depth; i--)
		{
			dim3 threads(min(512u, 1u << (3u * i)));
			__mipmap_1_layer<<<((uint)ceilf((1u << (3u * i)) / (float)threads.x)), threads>>>(octree.gpu_buffer_ptr, i);
		}
		__compute_barnes_hut<<<((uint)ceilf(grid_side_length * grid_side_length * grid_side_length / (float)block_size_barnes_hut)), block_size_barnes_hut>>>(octree.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr); cuda_sync();
	}
};

#endif