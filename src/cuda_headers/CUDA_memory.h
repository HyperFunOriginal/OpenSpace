
#ifndef CUDA_MEM_H
#define CUDA_MEM_H

#include "cuda_runtime.h"
#include <string>
#include <stdexcept>

template <class T>
static cudaError_t cuda_alloc_buffer(T** buffer_ptr, const size_t& buffer_len)
{
    cudaError_t cudaStatus = cudaMalloc((void**)buffer_ptr, buffer_len * sizeof(T));;
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, ("Allocation of " + std::to_string(buffer_len) + "length buffer failed!\n").c_str());
    }
    return cudaStatus;
}

template <class T>
static cudaError_t cuda_copytogpu_buffer(const T* cpuB, T* gpuB, const size_t& buffer_len)
{
    cudaError_t cudaStatus = cudaMemcpy(gpuB, cpuB, buffer_len * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy to GPU memory.\n");
    }
    return cudaStatus;
}

template <class T>
static cudaError_t cuda_copyfromgpu_buffer(T* cpuB, const T* gpuB, const size_t& buffer_len)
{
    cudaError_t cudaStatus = cudaMemcpy(cpuB, gpuB, buffer_len * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy to CPU memory.\n");
    }
    return cudaStatus;
}

template <class T>
struct smart_cpu_buffer
{
    int temp_data;
    int dedicated_len;
    T* cpu_buffer_ptr;
    bool created;

    operator T* ()
    {
        if (!created)
            throw std::exception("Buffer already freed.");
        return cpu_buffer_ptr;
    }
    smart_cpu_buffer(int max_size) : dedicated_len(max_size), created(true), temp_data(0)
    {
        cpu_buffer_ptr = new T[max_size];
    }
    size_t total_size() const { return dedicated_len * sizeof(T); }

    void destroy()
    {
        delete[] cpu_buffer_ptr;
        temp_data = 0;
        created = false;
    }
};

template <class T>
struct smart_gpu_buffer
{
    int temp_data;
    size_t dedicated_len;
    T* gpu_buffer_ptr;
    bool created;

    size_t total_size() const { return dedicated_len * sizeof(T); }
    void swap_pointers(smart_gpu_buffer<T>& other)
    {
        if (!dedicated_len == other.dedicated_len)
            throw std::exception("Cannot swap buffers of differing lengths!");

        T* temp = gpu_buffer_ptr;
        gpu_buffer_ptr = other.gpu_buffer_ptr;
        other.gpu_buffer_ptr = temp;
    }

    operator T* ()
    {
        if (!created)
            throw std::exception("Buffer already freed or badly allocated.");
        return gpu_buffer_ptr;
    }
    smart_gpu_buffer() : dedicated_len(0), temp_data(0), created(false), gpu_buffer_ptr(nullptr)
    {

    }
    smart_gpu_buffer(size_t max_size) : dedicated_len(max_size), temp_data(0)
    {
        gpu_buffer_ptr = 0;
        if (cuda_alloc_buffer(&gpu_buffer_ptr, dedicated_len) != cudaSuccess)
            cudaFree(gpu_buffer_ptr);
        else
            created = true;
    }
    virtual void destroy()
    {
        if (created)
            cudaFree(gpu_buffer_ptr);
        created = false;
    }
};

template <class T>
struct smart_gpu_cpu_buffer : smart_gpu_buffer<T>
{
    T* cpu_buffer_ptr;

    smart_gpu_cpu_buffer(smart_cpu_buffer<T>& cpu_buffer, bool destroy_old_success, bool destroy_old_failure) : smart_gpu_buffer<T>(cpu_buffer.dedicated_len)
    {
        if (created)
        {
            cpu_buffer_ptr = cpu_buffer.cpu_buffer_ptr; 

            if (destroy_old_success)
            {
                cpu_buffer.destroy();
            }
        }
        else if (destroy_old_failure)
        {
            cpu_buffer.destroy();
        }
    }

    smart_gpu_cpu_buffer() : smart_gpu_buffer<T>()
    {

    }
    void swap_gpu_pointers(smart_gpu_buffer<T>& other)
    {
        if (!dedicated_len == other.dedicated_len)
            throw std::exception("Cannot swap buffers of differing lengths!");

        T* temp = gpu_buffer_ptr;
        gpu_buffer_ptr = other.gpu_buffer_ptr;
        other.gpu_buffer_ptr = temp;
    }
    smart_gpu_cpu_buffer(size_t max_size) : smart_gpu_buffer<T>(max_size)
    {
        if (created)
            cpu_buffer_ptr = new T[max_size];
    }

    cudaError_t copy_to_cpu()
    {
        return cuda_copyfromgpu_buffer<T>(cpu_buffer_ptr, gpu_buffer_ptr, dedicated_len);
    }
    cudaError_t copy_to_gpu()
    {
        return cuda_copytogpu_buffer<T>(cpu_buffer_ptr, gpu_buffer_ptr, dedicated_len);
    }

    void destroy() override
    {
        if (created)
        {
            cudaFree(gpu_buffer_ptr);
            delete[] cpu_buffer_ptr;
        }
        created = false;
    }
};


template <class T>
cudaError_t copy_to_cpu(const smart_gpu_buffer<T>& a, smart_cpu_buffer<T>& b)
{
    if (a.dedicated_len != b.dedicated_len)
        return cudaErrorInvalidHostPointer;
    return cuda_copyfromgpu_buffer<T>(b.cpu_buffer_ptr, a.gpu_buffer_ptr, a.dedicated_len);
}

template <class T>
cudaError_t copy_to_gpu(smart_gpu_buffer<T>& a, const smart_cpu_buffer<T>& b)
{
    if (a.dedicated_len != b.dedicated_len)
        return cudaErrorInvalidDevicePointer;
    return cuda_copytogpu_buffer<T>(b.cpu_buffer_ptr, a.gpu_buffer_ptr, a.dedicated_len);
}

template<typename... Args>
static cudaError_t cuda_invoke_kernel(void (*kernel) (Args...), const dim3& blocks, const dim3& threads, Args... args)
{
    kernel<<<blocks, threads>>> (args...);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
    }
    return cudaStatus;
}

template<typename... Args>
static cudaError_t cuda_invoke_kernel_sync(void (*kernel) (Args...), const dim3& blocks, const dim3& threads, Args... args)
{
    kernel <<<blocks, threads>>> (args...);
    cudaError_t cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernels!\n", cudaStatus);
    }
    return cudaStatus;
}

static cudaError_t cuda_sync()
{
    cudaError_t cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching kernels!\n", cudaStatus);
    }
    return cudaStatus;
}

#endif