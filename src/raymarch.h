#ifndef RAYMARCH_H
#define RAYMARCH_H
#include <iostream>
#include <fstream>
#include "lodepng.h"

template <class T>
void write_to_file(const smart_cpu_buffer<T>& arr, const char* filepath)
{
    std::ofstream aa(filepath, std::ofstream::binary);
    const char* ptr = reinterpret_cast<const char*>(arr.cpu_buffer_ptr);
    aa.write(ptr, arr.dedicated_len * sizeof(T));
    aa.close();
}

template <class T>
void write_to_file(const smart_gpu_cpu_buffer<T>& arr, const char* filepath)
{
    std::ofstream aa(filepath, std::ofstream::binary);
    const char* ptr = reinterpret_cast<const char*>(arr.cpu_buffer_ptr);
    aa.write(ptr, arr.dedicated_len * sizeof(T));
    aa.close();
}

template <class T>
void read_from_file(smart_cpu_buffer<T>& arr, const char* filepath)
{
    std::ifstream file(filepath, std::ifstream::binary);
    char* ptr = reinterpret_cast<char*>(arr.cpu_buffer_ptr);

    file.seekg(0, std::ios::end);
    size_t len = file.tellg();
    file.seekg(0, file.beg);

    file.read(ptr, min(len, arr.dedicated_len * sizeof(T)));
}

template <class T>
void read_from_file(smart_gpu_cpu_buffer<T>& arr, const char* filepath)
{
    std::ifstream file(filepath, std::ifstream::binary);
    char* ptr = reinterpret_cast<char*>(arr.cpu_buffer_ptr);

    file.seekg(0, std::ios::end);
    size_t len = file.tellg();
    file.seekg(0, file.beg);

    file.read(ptr, min(len, arr.dedicated_len * sizeof(T)));
}

#include <windows.h>
bool create_folder(const char* path)
{
    return CreateDirectory(path, NULL) || ERROR_ALREADY_EXISTS == GetLastError();
}

inline __host__ __device__ uint ___rgba(const float4 val)
{
    if (isnan(val.x) || isnan(val.y) || isnan(val.z) || isnan(val.w))
        return 4294902015u;
    const uint4 v = make_uint4(clamp(val, 0.f, 1.f) * 255.f);
    return (v.w << 24) | (v.z << 16) | (v.y << 8) | v.x;
}

inline __host__ __device__ float4 ___hue_svalue(float2 HsV)
{
    if (isnan(HsV.x))
        HsV = make_float2(0.f);
    float3 col = clamp(make_float3(cosf(HsV.x) + .5f, cosf(HsV.x - 2.09439510239f) + .5f, cosf(HsV.x + 2.09439510239f) + .5f), 0.f, 1.f);
    return make_float4(lerp(col, make_float3(HsV.y > 1.f), (HsV.y > 1.f) ? (1.f - 1.f / HsV.y) : 1.f - cbrtf(HsV.y)), 1.f);
}

#include "gravitation.h"

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
    return isfinite(closest) ? closest : nan(0);
}


// Temporary debug view
__global__ void ___write_image_octree(uint* pixels, const grid_cell_ensemble* cells, const uint width, const uint height)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= width || idx.y >= height)
        return;

    uint coords = idx.y * width + idx.x;
    uint morton_index = __morton_index(make_float3(idx.x * domain_size_km / (float)width, idx.y * domain_size_km / (float)height, 0.f));
    float pseudo_depth = 0.f;

    for (uint i = morton_index, float closeness = .5f / grid_side_length; i < grid_cell_count; i = add_morton_indices(i, 4u), closeness += 1.f / grid_side_length)
    {
        float display_depth = 2.f * closeness - 1.f; display_depth *= 1.11803398875f * rsqrtf(1.f + display_depth * display_depth * 4.f); display_depth += .5f;
        pseudo_depth = lerp(display_depth * 1.25f, pseudo_depth, expf(-cells[__octree_depth_index(grid_dimension_pow) + i].total_mass_Tg / (size_grid_cell_km * size_grid_cell_km * size_grid_cell_km * 4E+4f)));
    }

    pixels[coords] = ___rgba(make_float4(pseudo_depth, pseudo_depth, pseudo_depth, 1.f));
}

void save_octree_image(smart_gpu_cpu_buffer<uint>& temp, const gravitational_simulation& simulation, const int width, const int height, const char* filename)
{
    const dim3 threads(min(width, 16), min(height, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    ___write_image_octree<<<blocks, threads>>>(temp.gpu_buffer_ptr, simulation.octree.gpu_buffer_ptr, width, height);
    temp.copy_to_cpu(); cuda_sync(); lodepng_encode32_file(filename, reinterpret_cast<const unsigned char*>(temp.cpu_buffer_ptr), width, height);
}


struct camera
{
    float3 position;
    float3 direction_right;
    float3 direction_up;
    float exposure;

    __host__ void set_fov(float fov_horizontal, float2 aspect_ratio)
    {
        fov_horizontal = tanf(fov_horizontal);
        direction_right *= fov_horizontal / length(direction_right);
        direction_up *= fov_horizontal * aspect_ratio.x / (length(direction_up) * aspect_ratio.y);
    }
    __host__ __device__ float2 fov() const {
        return make_float2(atanf(length(direction_right)), atanf(length(direction_up)));
    }
    __host__ __device__ float3 direction_fwd() const {
        return normalize(cross(direction_right, direction_up));
    }
    __host__ __device__ float3 direction(const uint2 idx, const uint width, const uint height) const {
        return normalize(direction_fwd() + ((idx.x * 2.f / (float)width) - 1.f) * direction_right + ((idx.y * 2.f / (float)height) - 1.f) * direction_up);
    }

    __host__ camera(float3 position, float3 direction_right, float3 direction_up, float exposure) : position(position), direction_right(direction_right), direction_up(direction_up), exposure(exposure)
    {
        float original_len = length(direction_up);
        direction_up = normalize(direction_up - dot(direction_right, direction_up) / dot(direction_right, direction_right) * direction_right) * original_len;
    }
};

// Broken as of now
// Implicit octree with morton indexing and a grid of sidelength 2^N, then raymarching straight through grids or collections thereof if there are no points there, i.e. cumsum over a certain patch is not increasing 
template <class T>
__global__ void ___write_image_raymarching(uint* pixels, const uint* cell_bounds, const uint width, const uint height, const T renderer, const camera cam)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= width || idx.y >= height) { return; }
    float3 ray_position = cam.position, ray_dir = cam.direction(idx, width, height);

    uint depth = 0u; uint iter_count = 0u;
    float4 col = make_float4(0.f, 0.3f, .6f, 1.f);

    if (global_min(cam.position) < 0.f || global_max(cam.position) > domain_size_km)
    {
        float dist_step = __ray_cube_intersection(ray_dir, ray_position, domain_size_km);
        if (isnan(dist_step)) { goto RETURN; }
        ray_position += (domain_size_km * 4E-4f + dist_step) * ray_dir;
    }

    while (iter_count < 500u)
    {
    RESET:
        iter_count++;
        const uint morton_index = __morton_index(ray_position);
        while (__count_particles(cell_bounds, morton_index, depth) == 0u) // Coarsen the view until it can't
        {
            if (depth == 0u) { goto RETURN; }
            depth--;
        }
        while (__count_particles(cell_bounds, morton_index, depth) != 0u && depth <= grid_dimension_pow) // Refine until it doesn't need to
            depth++;

        if (depth > grid_dimension_pow) // If at the most refined level, iterate through particles
        {
            const float3 grid_cell_corner_pos = floorf(ray_position / size_grid_cell_km) * size_grid_cell_km;
            const uint start_idx = __read_start_idx(cell_bounds, morton_index), end_idx = __read_end_idx(cell_bounds, morton_index);

            while (global_min(ray_position - grid_cell_corner_pos) >= 0.f && global_max(ray_position - grid_cell_corner_pos) <= size_grid_cell_km)
            {
                float closest_dst = INFINITY; uint closest_idx = 0u;
                renderer.closest_index_particle(closest_dst, closest_idx, ray_position, start_idx, end_idx);
                float4 tgt_col = renderer.apply_colour_intersect(ray_position, ray_dir, closest_idx);
                col.x = lerp(col.x, tgt_col.x, tgt_col.w * col.w);
                col.y = lerp(col.y, tgt_col.y, tgt_col.w * col.w);
                col.z = lerp(col.z, tgt_col.z, tgt_col.w * col.w);
                col.w *= 1.f - tgt_col.w;
                if (col.w < 1E-4f) { goto RETURN; }
                ray_position += closest_dst * ray_dir; // sphere-tracing
            }
            depth = grid_dimension_pow;
            goto RESET;
        }
        // found the largest empty cell, march through entire cell.
        const float cube_size = domain_size_km / (1u << depth);
        float dist_step = __ray_cube_intersection(ray_dir, ray_position - floorf(ray_position / cube_size) * cube_size, cube_size);
        if (isnan(dist_step)) { goto RETURN; }
        ray_position += (domain_size_km * 4E-4f + dist_step) * ray_dir;
    }

RETURN:
    col *= cam.exposure;
    pixels[idx.y * width + idx.x] = ___rgba(make_float4(col.x, col.y, col.z, 1.f));
}

// Broken as of now
template <class T>
void raymarch_render(smart_gpu_cpu_buffer<uint>& temp, const spatial_grid& simulation, const camera& cam, const T& renderer, const int width, const int height, const char* filename)
{
    const dim3 threads(min(width, 16), min(height, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    ___write_image_raymarching<<<blocks, threads>>>(temp.gpu_buffer_ptr, simulation.cell_bounds.gpu_buffer_ptr, width, height, renderer, cam);
    temp.copy_to_cpu(); cuda_sync(); lodepng_encode32_file(filename, reinterpret_cast<const unsigned char*>(temp.cpu_buffer_ptr), width, height);
}

#endif