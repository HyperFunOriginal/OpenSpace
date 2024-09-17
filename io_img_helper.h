#ifndef IO_HELPER_H
#define IO_HELPER_H

#include <stdio.h>
#include <iostream>
#include <fstream>
#include "CUDA_memory.h"

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

#endif