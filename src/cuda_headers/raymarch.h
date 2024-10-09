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
// Temporary debug view
/*__global__ void ___write_image_octree(uint* pixels, const grid_cell_ensemble* cells, const uint width, const uint height)
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
        float optical_thickness = cells[__octree_depth_index(grid_dimension_pow) + i].total_mass_Tg / (size_grid_cell_km * size_grid_cell_km * size_grid_cell_km * 1E+3f);
        pseudo_depth = lerp(closeness, pseudo_depth, expf(-optical_thickness * size_grid_cell_km * 1E-2f));
    }
    pixels[coords] = ___rgba(make_float4(pseudo_depth, pseudo_depth, pseudo_depth, 1.f));
}*/
__global__ void ___write_image_octree(uint* pixels, const grid_cell_ensemble* cells, const uint width, const uint height)
{
    const uint2 idx = make_uint2(threadIdx + blockDim * blockIdx);
    if (idx.x >= width || idx.y >= height)
        return;

    uint coords = idx.y * width + idx.x;
    uint morton_index = __morton_index(make_float3(idx.x * domain_size_km / (float)width, idx.y * domain_size_km / (float)height, 0.f));
    float render = 0.f;

    for (uint i = morton_index; i < grid_cell_count; i = add_morton_indices(i, 4u))
    {
        float optical_thickness = cells[__octree_depth_index(grid_dimension_pow) + i].total_mass_Tg / (size_grid_cell_km * size_grid_cell_km * size_grid_cell_km * 1.5E+4f);
        render = fmaxf(render, optical_thickness);
    }
    render = cbrtf(render);
    pixels[coords] = ___rgba(make_float4(render, render, render, 1.f));
}
void save_octree_image(smart_gpu_cpu_buffer<uint>& temp, const gravitational_simulation& simulation, const int width, const int height, const char* filename)
{
    const dim3 threads(min(width, 16), min(height, 16));
    const dim3 blocks((uint)ceilf(width / (float)threads.x), (uint)ceilf(height / (float)threads.y));
    ___write_image_octree<<<blocks, threads>>>(temp.gpu_buffer_ptr, simulation.octree.gpu_buffer_ptr, width, height);
    temp.copy_to_cpu(); cuda_sync(); lodepng_encode32_file(filename, reinterpret_cast<const unsigned char*>(temp.cpu_buffer_ptr), width, height);
}



#endif