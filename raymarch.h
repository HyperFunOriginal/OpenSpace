#ifndef RAYMARCH_H
#define RAYMARCH_H
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "spatial_grid.h"

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


#endif