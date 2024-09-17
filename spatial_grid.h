#ifndef SPATIAL_GRID_H
#define SPATIAL_GRID_H
#include "cum_sum.h"

__device__ constexpr uint minimum_depth = 1u;
__device__ constexpr uint grid_dimension_pow = 6u;
__device__ constexpr float domain_size_km = 100000.f;
__device__ constexpr uint grid_side_length = 1u << grid_dimension_pow;
__device__ constexpr float size_grid_cell_km = domain_size_km / grid_side_length;

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
	__device__ __host__ void set_true_pos(const float3 pos)
	{
		position = make_uint3(clamp(pos / domain_size_km, 0.f, 1.f) * 4294967040.f);
	}
};
static_assert(sizeof(particle) == 16, "Wrong size!");



__device__ __host__ float3 __cell_pos_from_index(uint morton, uint depth) {
	uint3 position = make_uint3(0u);
	for (uint i = 0u; i < depth; i++)
	{
		position.x |= ((morton >> (3u * i	  )) & 1u) << i;
		position.y |= ((morton >> (3u * i + 1u)) & 1u) << i;
		position.z |= ((morton >> (3u * i + 2u)) & 1u) << i;
	}
	return (make_float3(position) + .5f) * domain_size_km / (1u << depth);
}

__device__ __host__ uint __morton_index(const float3 pos) {
	uint3 position = make_uint3(clamp(pos / domain_size_km, 0.f, 1.f) * 4294967040.f); uint result = 0u;
	for (uint i = 0u; i < grid_dimension_pow; i++)
	{
		result |= ((position.x >> (i + 32u - grid_dimension_pow)) & 1u) << (3u * i) |
			((position.y >> (i + 32u - grid_dimension_pow)) & 1u) << (3u * i + 1u) |
			((position.z >> (i + 32u - grid_dimension_pow)) & 1u) << (3u * i + 2u);
	}
	return result;
}

__device__ __host__ uint __read_start_idx(const uint* cell_pos, const uint morton_index)
{
	return (morton_index == 0u) ? 0u : cell_pos[morton_index - 1u];
}

__device__ __host__ uint __count_particles(const uint* cell_pos, uint morton_index, uint depth)
{
	depth = (grid_dimension_pow - depth) * 3u; morton_index >>= depth;
	return __read_start_idx(cell_pos, (morton_index + 1u) << depth) - __read_start_idx(cell_pos, morton_index << depth);
}

__device__ constexpr uint mask_morton_x = 0b01001001001001001001001001001001u;
__device__ constexpr uint mask_morton_y = 0b10010010010010010010010010010010u;
__device__ constexpr uint mask_morton_z = 0b00100100100100100100100100100100u;

__device__ __host__ uint add_morton_indices(uint morton_A, uint morton_B)
{
	uint x = (morton_A | (~mask_morton_x)) + (morton_B & mask_morton_x);
	uint y = (morton_A | (~mask_morton_y)) + (morton_B & mask_morton_y);
	uint z = (morton_A | (~mask_morton_z)) + (morton_B & mask_morton_z);
	return (x & mask_morton_x) | (y & mask_morton_y) | (z & mask_morton_z);
}

/// <summary>
/// Yields if morton index lies on the boundary of: [-z][-y][-x][+z][+y][+x]; Big Endian.
/// </summary>
/// <param name="morton">: Morton index</param>
/// <param name="depth">: Depth to check for</param>
/// <returns></returns>
__device__ __host__ uint bounds(uint morton, uint depth)
{
	morton >>= 3u * (grid_dimension_pow - depth);
	uint result = ((morton & mask_morton_x) == 0) | (((morton & mask_morton_y) == 0) << 1u) | (((morton & mask_morton_z) == 0) << 2u);
	morton ^= (~0u) >> (32u - depth * 3u);
	return (result << 3u) | ((morton & mask_morton_x) == 0) | (((morton & mask_morton_y) == 0) << 1u) | (((morton & mask_morton_z) == 0) << 2u);
}


template <class T>
__global__ void __set_empty(T* cell_counts, uint number_cells)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= number_cells) { return; }
	cell_counts[idx] = T();
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
	new_buffer[__read_start_idx(cell_pos, part.morton_index()) + part.in_cell_index()] = old_buffer[idx];
}

__global__ void __copy_spatial_counting_sort(const uint* cell_pos, const particle* old_buffer, particle* new_buffer, uint number_particles)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= number_particles) { return; }

	particle part = old_buffer[idx];
	new_buffer[__read_start_idx(cell_pos, part.morton_index()) + part.in_cell_index()] = part;
}

template <class T>
struct particle_data_buffer
{
	smart_gpu_cpu_buffer<T> buffer;
	smart_gpu_buffer<T> temp;
	particle_data_buffer(size_t dedicated_len) : buffer(dedicated_len), temp(dedicated_len) {}
	void swap_pointers() { buffer.swap_gpu_pointers(temp); }
	void destroy() { buffer.destroy(); temp.destroy(); }
	void copy_to_cpu() { buffer.copy_to_cpu(); }
	void copy_to_gpu() { buffer.copy_to_gpu(); }
};

struct spatial_grid
{
	size_t particle_capacity;
	particle_data_buffer<particle> particles;
	smart_gpu_buffer<uint> cell_bounds;
public:
	/// <summary>
	/// Copy data associated with particles in sort. Override for required data.
	/// </summary>
	/// <param name="cell_bounds">: Cell boundaries.</param>
	/// <param name="targets">: Target particles.</param>
	virtual void counting_sort_transfers(const smart_gpu_buffer<uint>& cell_bounds, const smart_gpu_buffer<particle>& targets) {}
	spatial_grid(size_t allocation_particles) : particles(allocation_particles), particle_capacity(allocation_particles), cell_bounds(grid_side_length* grid_side_length* grid_side_length) {
		dim3 threads(particles.buffer.dedicated_len > 512u ? 512u : particles.buffer.dedicated_len);
		dim3 blocks((uint)ceilf(particles.buffer.dedicated_len / (float)threads.x));

		__set_empty<<<threads, blocks>>>(particles.buffer.gpu_buffer_ptr, particles.buffer.dedicated_len);
	}
	/// <summary>
	/// Sorts spatially via morton indices the order of particles stored in particles buffer.
	/// </summary>
	void sort_spatially()
	{
		dim3 threads(particles.buffer.dedicated_len > 512u ? 512u : particles.buffer.dedicated_len);
		dim3 blocks((uint)ceilf(particles.buffer.dedicated_len / (float)threads.x));

		__set_empty<<<threads, blocks>>>(particles.temp.gpu_buffer_ptr, particles.temp.dedicated_len);
		__set_empty<<<((uint)ceilf(cell_bounds.dedicated_len / 512.f)), 512u>>>(cell_bounds.gpu_buffer_ptr, cell_bounds.dedicated_len);
	
		__locate_in_cells<<<blocks, threads>>>(cell_bounds.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particles.buffer.dedicated_len);
		apply_incl_cum_sum<uint>(cell_bounds); counting_sort_transfers(cell_bounds, particles.buffer); cuda_sync();

		__copy_spatial_counting_sort<<<blocks, threads>>>(cell_bounds.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, particles.temp.gpu_buffer_ptr, particles.buffer.dedicated_len);
		cuda_sync(); particles.swap_pointers();
	}
};

template <class T>
void counting_sort_data_transfer(const smart_gpu_buffer<uint>& cell_bounds, const smart_gpu_buffer<particle>& targets, particle_data_buffer<T>& buffer)
{
	dim3 threads(targets.dedicated_len > 512u ? 512u : targets.dedicated_len);
	dim3 blocks((uint)ceilf(targets.dedicated_len / (float)threads.x));
	__copy_spatial_counting_sort_data<T> << <blocks, threads >> > (cell_bounds.gpu_buffer_ptr, targets.gpu_buffer_ptr, buffer.buffer.gpu_buffer_ptr, buffer.temp.gpu_buffer_ptr, targets.dedicated_len);
	buffer.swap_pointers();
}

#endif