#ifndef SIM_HELPER_H
#define SIM_HELPER_H

#include "CUDA_memory.h"
#include "helper_math.h"

// Tunable
__device__ constexpr bool wrap_around = false;
__device__ constexpr uint grid_dimension_pow = 8u; // Critical; Must be sufficiently large for moderate particle per cell counts. Must not be too large such that hydrodynamic averaging remains correct.
__device__ constexpr float domain_size_km = 37000.f;
__device__ constexpr uint minimum_depth = 1u; // Already optimal

// Derived
__device__ constexpr uint grid_side_length = 1u << grid_dimension_pow;
__device__ constexpr uint grid_cell_count = 1u << (3u * grid_dimension_pow);
__device__ constexpr float size_grid_cell_km = domain_size_km / grid_side_length;

//////////////////////////////////
////	  Morton Indexing	  ////
//////////////////////////////////

__device__ constexpr uint mask_morton_x = 0b01001001001001001001001001001001u;
__device__ constexpr uint mask_morton_y = 0b10010010010010010010010010010010u;
__device__ constexpr uint mask_morton_z = 0b00100100100100100100100100100100u;

__device__ __host__ float3 __cell_pos_from_index(uint morton, uint depth) {
	uint3 position = make_uint3(0u);
	for (uint i = 0u; i < depth; i++)
	{
		position.x |= ((morton >> (3u * i)) & 1u) << i;
		position.y |= ((morton >> (3u * i + 1u)) & 1u) << i;
		position.z |= ((morton >> (3u * i + 2u)) & 1u) << i;
	}
	return (make_float3(position) + .5f) * domain_size_km / (1u << depth);
}
__device__ __host__ uint __morton_index(const float3 pos) {
	uint3 position = make_uint3(clamp(pos / domain_size_km, 0.f, 0.999999940395355224609375f) * 4294967296.f); uint result = 0u;
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
__device__ __host__ uint __read_end_idx(const uint* cell_pos, const uint morton_index)
{
	return cell_pos[morton_index];
}
__device__ __host__ uint __count_particles(const uint* cell_pos, uint morton_index, uint depth)
{
	depth = (grid_dimension_pow - depth) * 3u; morton_index >>= depth;
	return __read_start_idx(cell_pos, (morton_index + 1u) << depth) - __read_start_idx(cell_pos, morton_index << depth);
}

__device__ __host__ uint sub_morton_indices(uint morton_A, uint morton_B)
{
	uint x = (morton_A & mask_morton_x) - (morton_B & mask_morton_x);
	uint y = (morton_A & mask_morton_y) - (morton_B & mask_morton_y);
	uint z = (morton_A & mask_morton_z) - (morton_B & mask_morton_z);
	return (x & mask_morton_x) | (y & mask_morton_y) | (z & mask_morton_z);
}
__device__ __host__ uint add_morton_indices(uint morton_A, uint morton_B)
{
	uint x = (morton_A | (~mask_morton_x)) + (morton_B & mask_morton_x);
	uint y = (morton_A | (~mask_morton_y)) + (morton_B & mask_morton_y);
	uint z = (morton_A | (~mask_morton_z)) + (morton_B & mask_morton_z);
	return (x & mask_morton_x) | (y & mask_morton_y) | (z & mask_morton_z);
}

//////////////////////////////////
//// Rendering Helper Methods ////
//////////////////////////////////

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
inline __host__ __device__ float __ray_cube_intersection(float3 ray, float3 displacement_from_000_corner, const float cube_size)
{
	float3 ratios_l = -displacement_from_000_corner / ray;
	float3 ratios_r = (cube_size - displacement_from_000_corner) / ray;

	float closest = INFINITY;
	if (ratios_l.x >= 0.f)
	{
		float y = ray.y * ratios_l.x + displacement_from_000_corner.y;
		float z = ray.z * ratios_l.x + displacement_from_000_corner.z;
		if (fminf(y, z) >= 0.f && fmaxf(y, z) <= cube_size)
			closest = fminf(closest, ratios_l.x);
	}
	if (ratios_l.y >= 0.f)
	{
		float x = ray.x * ratios_l.y + displacement_from_000_corner.x;
		float z = ray.z * ratios_l.y + displacement_from_000_corner.z;
		if (fminf(x, z) >= 0.f && fmaxf(x, z) <= cube_size)
			closest = fminf(closest, ratios_l.y);
	}
	if (ratios_l.z >= 0.f)
	{
		float x = ray.x * ratios_l.z + displacement_from_000_corner.x;
		float y = ray.y * ratios_l.z + displacement_from_000_corner.y;
		if (fminf(x, y) >= 0.f && fmaxf(x, y) <= cube_size)
			closest = fminf(closest, ratios_l.z);
	}
	if (ratios_r.x >= 0.f)
	{
		float y = ray.y * ratios_r.x + displacement_from_000_corner.y;
		float z = ray.z * ratios_r.x + displacement_from_000_corner.z;
		if (fminf(y, z) >= 0.f && fmaxf(y, z) <= cube_size)
			closest = fminf(closest, ratios_r.x);
	}
	if (ratios_r.y >= 0.f)
	{
		float x = ray.x * ratios_r.y + displacement_from_000_corner.x;
		float z = ray.z * ratios_r.y + displacement_from_000_corner.z;
		if (fminf(x, z) >= 0.f && fmaxf(x, z) <= cube_size)
			closest = fminf(closest, ratios_r.y);
	}
	if (ratios_r.z >= 0.f)
	{
		float x = ray.x * ratios_r.z + displacement_from_000_corner.x;
		float y = ray.y * ratios_r.z + displacement_from_000_corner.y;
		if (fminf(x, y) >= 0.f && fmaxf(x, y) <= cube_size)
			closest = fminf(closest, ratios_r.z);
	}
	return isfinite(closest) ? closest : nanf("");
}

__device__ constexpr uint ___morton_offsets[3] = { 0u, 1u, 8u };
__device__ constexpr uint ___bitmasks_iterator[6] = { 0x04924924u, 0x070381c0u, 0x07fc0000u, 0x01249249u, 0x001c0e07u, 511u };
__device__ constexpr uint ___flags[64u] = { 1744838656u, 1610625024u, 1342186496u, 1207973376u, 536879120u, 402665496u, 134226962u, 13851u, 1744855040u, 1610641408u, 1342204928u, 1207991808u, 536895536u, 402681912u, 134245430u, 32319u, 1744904192u, 1610723328u, 1342252032u, 1208071680u, 536944784u, 402763992u, 134292626u, 112347u, 1745051648u, 1610870784u, 1342401536u, 1208221184u, 537092528u, 402911736u, 134442422u, 262143u, 1749032960u, 1616916480u, 1346905088u, 1215051264u, 541073424u, 408956952u, 138945554u, 7091739u, 1757437952u, 1625321472u, 1356360704u, 1224506880u, 549478448u, 417361976u, 148401206u, 16547391u, 1782652928u, 1667346432u, 1380525056u, 1265481216u, 574693520u, 459387096u, 172565650u, 57521883u, 1858297856u, 1742991360u, 1457220608u, 1342176768u, 650338736u, 535032312u, 249261494u, 134217727u };

#define FOREACH(type, v, iter) for(type v = iter.yield(true); v != iter.end(); v = iter.yield(false))
struct morton_cell_iterator
{
	uint bitmask, morton;

	constexpr __device__ __host__ uint end() const {
		return ~0u;
	}

	/// <summary>
	/// Yields the next morton index. Returns UINT_MAX if the iterator has reached the end.
	/// </summary>
	/// <returns></returns>
	__device__ __host__ uint yield(bool reset)
	{
		uint current_idx;
		if (reset) { bitmask &= 0x07ffffffu; }
		do
		{
			current_idx = bitmask >> 27u;
			if (current_idx >= 27u) { return end(); }
			bitmask = (bitmask & 0x07ffffffu) | ((current_idx + 1u) << 27u);
		} while ((bitmask & (1u << current_idx)) == 0u);

		return add_morton_indices(morton, ___morton_offsets[current_idx % 3u]
			| (___morton_offsets[(current_idx / 3u) % 3u] << 1u)
			| (___morton_offsets[current_idx / 9u] << 2u));
	}
	__device__ __host__ morton_cell_iterator() : bitmask(), morton() {}
	__device__ __host__ morton_cell_iterator(uint center_morton_idx) : bitmask(0x07ffffffu), morton(center_morton_idx)
	{
		morton = sub_morton_indices(center_morton_idx, 7u);
		uint temp = bounds(center_morton_idx, grid_dimension_pow);
#pragma unroll
		for (uint i = 0; i < 6; i++)
			bitmask &= ~(___bitmasks_iterator[i] * ((temp & (1u << i)) != 0u));
	}
	__device__ __host__ morton_cell_iterator(uint center_morton_idx, float3 displacement_from_000_corner, uint depth, float tolerance) : bitmask(0u), morton(center_morton_idx)
	{
		displacement_from_000_corner *= (1u << depth) / domain_size_km;
		uint tolerance_check = (displacement_from_000_corner.x < tolerance) | ((displacement_from_000_corner.y < tolerance) << 1u) | ((displacement_from_000_corner.z < tolerance) << 2u);
		tolerance = 1.f - tolerance; tolerance_check |= ((displacement_from_000_corner.x > tolerance) << 3u) | ((displacement_from_000_corner.y > tolerance) << 4u) | ((displacement_from_000_corner.z > tolerance) << 5u);

		bitmask = ___flags[tolerance_check];
		morton = sub_morton_indices(center_morton_idx, 7u);
		if (tolerance_check != 0u)
		{
			uint temp = bounds(center_morton_idx, depth);
#pragma unroll
			for (uint i = 0; i < 6; i++)
				bitmask &= ~(___bitmasks_iterator[i] * ((temp & (1u << i)) != 0u));
		}
	}
};
static_assert(sizeof(morton_cell_iterator) == 8, "Wrong Size!");

#endif