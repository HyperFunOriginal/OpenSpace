#ifndef GRAVITATION_H
#define GRAVITATION_H
#include "kinematics.h"

// Tunable
__device__ constexpr bool compute_fast_approx = true;
__device__ constexpr bool compute_gravity_empty_cell = false;
__device__ constexpr float barnes_hut_criterion = 0.35f;
__device__ constexpr float G_in_Tg_km_units = 6.6743015E-8f;

struct grid_cell_ensemble
{
	float standard_radius_km;
	float total_mass_Tg;
	float3 deviatoric_pos_km;
	float3 average_acceleration_ms2;
	__device__ __host__ grid_cell_ensemble() : deviatoric_pos_km(make_float3(0.f)), total_mass_Tg(1E-40f), standard_radius_km(0.f) {}

	__device__ __host__ float average_density_kgm3() const { return total_mass_Tg / fmaxf(4.18879020479f * standard_radius_km * standard_radius_km * standard_radius_km, 1E-5f); }
};

////////////////////////////////////
//// Gravitational Acceleration ////
////////////////////////////////////

// inexact approximation
__device__ __host__ float grav_acc_factor_fast(float separation, float standard_deviation)
{
	standard_deviation *= 1.41421356237f;
	return -1.f / (separation * separation * separation + standard_deviation * standard_deviation * standard_deviation * 1.35f);
}
// exact solution
__device__ __host__ float grav_acc_factor(float separation, float standard_deviation)
{
	standard_deviation *= 1.41421356237f;
	if (separation < .25f * standard_deviation)
		return -0.75225277806f / (standard_deviation * standard_deviation * standard_deviation);
	if (separation > 3.f * standard_deviation)
		return -1.f / (separation * separation * separation);
	return 1.1283791671f * expf(-separation * separation / (standard_deviation * standard_deviation)) / (separation * separation * standard_deviation) - erf_lossy(separation / standard_deviation) / (separation * separation * separation);
}

////////////////////////////////////
////  Implicit Octree Indexing  ////
////////////////////////////////////

__device__ __host__ constexpr uint __octree_depth_index(uint depth)
{
	return (mask_morton_x & ((~0u) >> (32u - depth * 3u))) - (mask_morton_x & ((~0u) >> (32u - minimum_depth * 3u)));
}

////////////////////////////////////
////	  Ensemble Kernels	    ////
////////////////////////////////////

__global__ void __average_ensemble(const particle_kinematics* kinematics, const particle* positions, const uint* cell_pos, grid_cell_ensemble* grid_cells)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= grid_cell_count) { return; }
	const uint start_pos = __read_start_idx(cell_pos, idx), end_pos = __read_end_idx(cell_pos, idx);
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
	grid_cells[idx + __octree_depth_index(grid_dimension_pow)] = this_ensemble;
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
		grid_cell_ensemble target = grid_cells[i + __octree_depth_index(target_depth + 1)];
		float3 rel_pos = target.deviatoric_pos_km + __cell_pos_from_index(i, target_depth + 1) - this_cell_pos;

		this_ensemble.total_mass_Tg += target.total_mass_Tg;
		this_ensemble.deviatoric_pos_km += rel_pos * target.total_mass_Tg;
		variances += (rel_pos * rel_pos + target.standard_radius_km * target.standard_radius_km * .333333333f) * target.total_mass_Tg;
	}

	this_ensemble.deviatoric_pos_km /= this_ensemble.total_mass_Tg;
	variances = (variances / this_ensemble.total_mass_Tg) - this_ensemble.deviatoric_pos_km * this_ensemble.deviatoric_pos_km;
	this_ensemble.standard_radius_km = sqrtf(variances.x + variances.y + variances.z);
	grid_cells[idx + __octree_depth_index(target_depth)] = this_ensemble;
}

////////////////////////////////////
////  Gravitation and Dynamics	////
////////////////////////////////////
__global__ void __compute_barnes_hut(grid_cell_ensemble* octree, const uint* cell_bounds)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= grid_cell_count) { return; }
	float3 acceleration_ms2 = make_float3(0.f);

	if (compute_gravity_empty_cell || __count_particles(cell_bounds, idx, grid_dimension_pow) > 0) {
		float3 this_position_km = octree[idx + __octree_depth_index(grid_dimension_pow)].deviatoric_pos_km + __cell_pos_from_index(idx, grid_dimension_pow);
		uint depth = minimum_depth; uint morton_index = 0u;

		while (morton_index < (1u << (depth * 3u))) // implicit octree depth-first traversal
		{
			const grid_cell_ensemble other_c = octree[morton_index + __octree_depth_index(depth)];
			float3 separation = this_position_km - (other_c.deviatoric_pos_km + __cell_pos_from_index(morton_index, depth));
			if ((other_c.standard_radius_km * other_c.standard_radius_km / dot(separation, separation) > barnes_hut_criterion * barnes_hut_criterion) && (depth < grid_dimension_pow))
			{
				depth++;
				morton_index <<= 3u;
				continue;
			}
			acceleration_ms2 += separation * (other_c.total_mass_Tg * G_in_Tg_km_units *
				(compute_fast_approx ? grav_acc_factor_fast(length(separation) + 1E-10f, other_c.standard_radius_km + 1E-10f) : grav_acc_factor(length(separation) + 1E-10f, other_c.standard_radius_km + 1E-10f)));
			if ((morton_index & 7u) == 7u && depth > 0)
			{
				morton_index >>= 3u;
				depth--;
			}
			morton_index++;
		}
	}
	octree[idx + __octree_depth_index(grid_dimension_pow)].average_acceleration_ms2 = acceleration_ms2;
}
__global__ void __apply_gravitation(const grid_cell_ensemble* octree, const uint* cell_bounds, const particle* particles, particle_kinematics* kinematics, const uint particle_capacity)
{
	uint idx = threadIdx.x + blockDim.x * blockIdx.x;
	if (idx >= particle_capacity) { return; }

	if (!particles[idx].exists()) { return; }
	int3 this_pos = make_int3(particles[idx].position >> 1u);
	uint morton_index = particles[idx].morton_index();
	float3 acceleration_ms2 = octree[morton_index + __octree_depth_index(grid_dimension_pow)].average_acceleration_ms2;

	for (uint i = __read_start_idx(cell_bounds, morton_index), j = __read_end_idx(cell_bounds, morton_index); i < j; i++)
	{
		if (i == idx) { continue; }
		float3 separation = make_float3(this_pos - make_int3(particles[i].position >> 1u)) * (domain_size_km / 2147483648.f); // greater precision in int than float; 31 bits vs 23 bits
		acceleration_ms2 += separation * (kinematics[i].mass_Tg * G_in_Tg_km_units *
			(compute_fast_approx ? grav_acc_factor_fast(length(separation) + 1E-10f, kinematics[i].radius_km + 1E-10f) : grav_acc_factor(length(separation) + 1E-10f, kinematics[i].radius_km + 1E-10f)));
	}

	kinematics[idx].acceleration_ms2 = acceleration_ms2;
}

//////////////////////////////////
////	  Main Structures	  ////
//////////////////////////////////

struct gravitational_simulation : kinematic_simulation
{
	smart_gpu_buffer<grid_cell_ensemble> octree;
private:
	smart_cpu_buffer<grid_cell_ensemble> coarsest;
public:

	gravitational_simulation(size_t allocation_particles) : kinematic_simulation(allocation_particles), octree(__octree_depth_index(grid_dimension_pow + 1)), coarsest(1u << (3u * minimum_depth)) { }
	void generate_gravitational_data()
	{
		__average_ensemble<<<((uint)ceilf(cell_bounds.dedicated_len / 512.f)), 512u>>>(kinematic_data.buffer.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr, octree.gpu_buffer_ptr);
		for (int i = grid_dimension_pow - 1; i >= minimum_depth; i--)
		{
			dim3 threads(min(512u, 1u << (3u * i)));
			__mipmap_1_layer<<<((uint)ceilf((1u << (3u * i)) / (float)threads.x)), threads>>>(octree.gpu_buffer_ptr, i);
		}
		__compute_barnes_hut<<<((uint)ceilf(grid_cell_count / 512.f)), 512u>>>(octree.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr); cuda_sync();
	}
	void apply_kinematics_recenter(float timestep_s)
	{
		uint threads = particle_capacity < 512u ? particle_capacity : 512u;
		uint blocks = ceilf(particle_capacity / (float)threads);

		float3 avg_pos = make_float3(0.f);
		cudaMemcpy(coarsest.cpu_buffer_ptr, octree.gpu_buffer_ptr, (1u << (3u * minimum_depth)) * sizeof(grid_cell_ensemble), cudaMemcpyDeviceToHost);

		float total_mass = 1E-30f;
		for (uint i = 0u; i < 1u << (3u * minimum_depth); i++)
		{
			grid_cell_ensemble target = coarsest.cpu_buffer_ptr[i];
			avg_pos += (target.deviatoric_pos_km + __cell_pos_from_index(i, minimum_depth)) * target.total_mass_Tg;
			total_mass += target.total_mass_Tg;
		}
		avg_pos = (avg_pos / total_mass) - domain_size_km * .5f;

		__apply_kinematics<<<blocks, threads>>>(particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, timestep_s, particle_capacity, avg_pos);
		cuda_sync();
	}
	void apply_gravitation()
	{
		uint threads = particle_capacity < 512u ? particle_capacity : 512u;
		uint blocks = ceilf(particle_capacity / (float)threads);

		__apply_gravitation<<<blocks, threads>>>(octree.gpu_buffer_ptr, cell_bounds.gpu_buffer_ptr, particles.buffer.gpu_buffer_ptr, kinematic_data.buffer.gpu_buffer_ptr, particle_capacity);
	}
};

#endif