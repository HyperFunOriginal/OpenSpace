#ifndef PARALLEL_H
#define PARALLEL_H

#include "CUDA_memory.h"
#include "helper_math.h"

__device__ constexpr uint ____block_pow_cumsum = 9u;
__device__ constexpr uint ____block_size_cumsum = 1u << ____block_pow_cumsum;

template <class T>
__global__ void ____cum_sum_reduce(T* data, uint length, uint stride)
{
	__shared__ T shared[____block_size_cumsum]; uint internal_idx = threadIdx.x;
	uint grid_idx = (threadIdx.x + ____block_size_cumsum * blockIdx.x + 1u) * stride - 1u;
	shared[internal_idx] = (grid_idx < length) ? data[grid_idx] : T();

	for (uint i = 0u; i < ____block_pow_cumsum; i++)
	{
		internal_idx = (internal_idx << 1u) + 1u;
		__syncthreads();
		if (internal_idx < ____block_size_cumsum)
			shared[internal_idx] += shared[internal_idx - (1u << i)];
		__syncthreads();
	}
	if (grid_idx < length)
		data[grid_idx] = shared[threadIdx.x];
}

template <class T>
__global__ void ____cum_sum_post(T* data, uint length, uint stride)
{
	__shared__ T shared[____block_size_cumsum];
	uint internal_idx = ((threadIdx.x + 1u) << (____block_pow_cumsum - 1u)) - 1u;
	uint grid_idx = (threadIdx.x + ____block_size_cumsum * blockIdx.x + 1u) * stride - 1u;
	shared[threadIdx.x] = (grid_idx < length) ? data[grid_idx] : T();

	bool add = false; // propagate across blocks
	for (uint i = 0; i < ____block_pow_cumsum; i++)
		add |= (threadIdx.x + 1u) == (1 << i);
	if (blockIdx.x > 0 && add)
		shared[threadIdx.x] += data[____block_size_cumsum * blockIdx.x * stride - 1u];

	for (int i = ____block_pow_cumsum - 2; i >= 0; i--)
	{
		__syncthreads();
		if ((internal_idx + (1u << i)) < ____block_size_cumsum)
			shared[internal_idx + (1u << i)] += shared[internal_idx];
		__syncthreads();

		internal_idx >>= 1u;
	}
	if (grid_idx < length)
		data[grid_idx] = shared[threadIdx.x];
}

/// <summary>
/// Applies an inclusive cumulative sum to a GPU buffer of any relevant type. Runs in effective O(log n) time.
/// </summary>
/// <typeparam name="T">: Type of numeric object.</typeparam>
/// <param name="data">: GPU buffer</param>
/// <returns></returns>
template <class T>
void apply_incl_cum_sum(smart_gpu_buffer<T>& data)
{
	uint stride = 1u;
	while (stride < data.dedicated_len)
	{
		dim3 blocks(ceilf(data.dedicated_len / (float)(____block_size_cumsum * stride)));
		____cum_sum_reduce<<<blocks, ____block_size_cumsum>>>(data.gpu_buffer_ptr, data.dedicated_len, stride);
		stride <<= ____block_pow_cumsum;
	}
	stride >>= ____block_pow_cumsum;
	while (stride > 0)
	{
		dim3 blocks(ceilf(data.dedicated_len / (float)(____block_size_cumsum * stride)));
		____cum_sum_post<<<blocks, ____block_size_cumsum>>>(data.gpu_buffer_ptr, data.dedicated_len, stride);
		stride >>= ____block_pow_cumsum; 
	}
	cuda_sync();
}


template <class T>
inline __device__ __host__ T maximum_unified(T a, T b); // must be defined for your use case

template <class T>
inline __device__ __host__ T minimum_value_format(); // must be defined for your use case

template<>
inline __device__ __host__ uint maximum_unified(uint a, uint b)
{
	return max(a, b);
}
template<>
inline __device__ __host__ uint2 maximum_unified(uint2 a, uint2 b)
{
	return max(a, b);
}
template<>
inline __device__ __host__ uint3 maximum_unified(uint3 a, uint3 b)
{
	return max(a, b);
}
template<>
inline __device__ __host__ uint4 maximum_unified(uint4 a, uint4 b)
{
	return max(a, b);
}

template<>
inline __device__ __host__ uint minimum_value_format()
{
	return 0u;
}
template<>
inline __device__ __host__ uint2 minimum_value_format()
{
	return make_uint2(0u);
}
template<>
inline __device__ __host__ uint3 minimum_value_format()
{
	return make_uint3(0u);
}
template<>
inline __device__ __host__ uint4 minimum_value_format()
{
	return make_uint4(0u);
}

template<>
inline __device__ __host__ int maximum_unified(int a, int b)
{
	return max(a, b);
}
template<>
inline __device__ __host__ int2 maximum_unified(int2 a, int2 b)
{
	return max(a, b);
}
template<>
inline __device__ __host__ int3 maximum_unified(int3 a, int3 b)
{
	return max(a, b);
}
template<>
inline __device__ __host__ int4 maximum_unified(int4 a, int4 b)
{
	return max(a, b);
}

template<>
inline __device__ __host__ int minimum_value_format()
{
	return (1<<31);
}
template<>
inline __device__ __host__ int2 minimum_value_format()
{
	return make_int2(1 << 31);
}
template<>
inline __device__ __host__ int3 minimum_value_format()
{
	return make_int3(1 << 31);
}
template<>
inline __device__ __host__ int4 minimum_value_format()
{
	return make_int4(1 << 31);
}

template<>
inline __device__ __host__ float maximum_unified(float a, float b)
{
	return fmaxf(a, b);
}
template<>
inline __device__ __host__ float2 maximum_unified(float2 a, float2 b)
{
	return fmaxf(a, b);
}
template<>
inline __device__ __host__ float3 maximum_unified(float3 a, float3 b)
{
	return fmaxf(a, b);
}
template<>
inline __device__ __host__ float4 maximum_unified(float4 a, float4 b)
{
	return fmaxf(a, b);
}

template<>
inline __device__ __host__ float minimum_value_format()
{
	return -INFINITY;
}
template<>
inline __device__ __host__ float2 minimum_value_format()
{
	return make_float2(-INFINITY);
}
template<>
inline __device__ __host__ float3 minimum_value_format()
{
	return make_float3(-INFINITY);
}
template<>
inline __device__ __host__ float4 minimum_value_format()
{
	return make_float4(-INFINITY);
}

template <class T>
__global__ void ____maximum_reduce(T* data, uint length, uint stride)
{
	__shared__ T shared[____block_size_cumsum]; uint internal_idx = threadIdx.x;
	uint grid_idx = (threadIdx.x + ____block_size_cumsum * blockIdx.x) * stride;
	shared[internal_idx] = (grid_idx < length) ? data[grid_idx] : minimum_value_format<T>();

	for (uint i = 0u; i < ____block_pow_cumsum; i++)
	{
		internal_idx <<= 1u;
		__syncthreads();
		if (internal_idx < ____block_size_cumsum)
			shared[internal_idx] = maximum_unified<T>(shared[internal_idx], shared[internal_idx + (1u << i)]);
		__syncthreads();
	}
	if (grid_idx < length && threadIdx.x == 0u)
		data[grid_idx] = shared[threadIdx.x];
}
/// <summary>
/// Mutates the provided buffer to find the largest value therein. Runs in effective O(log n) time.
/// </summary>
/// <typeparam name="T">: Type of numeric object.</typeparam>
/// <param name="data">: GPU buffer</param>
/// <returns></returns>
template <class T>
T find_maximum_val(smart_gpu_buffer<T>& data)
{
	uint stride = 1u;
	while (stride < data.dedicated_len)
	{
		dim3 blocks(ceilf(data.dedicated_len / (float)(____block_size_cumsum * stride)));
		____maximum_reduce<<<blocks, ____block_size_cumsum>>>(data.gpu_buffer_ptr, data.dedicated_len, stride);
		stride <<= ____block_pow_cumsum; 
	}
	T result; cudaMemcpy(&result, data.gpu_buffer_ptr, sizeof(T), cudaMemcpyDeviceToHost);
	return result;
}

template <class T>
inline __device__ __host__ T minimum_unified(T a, T b); // must be defined for your use case

template <class T>
inline __device__ __host__ T maximum_value_format(); // must be defined for your use case

template<>
inline __device__ __host__ uint minimum_unified(uint a, uint b)
{
	return min(a, b);
}
template<>
inline __device__ __host__ uint2 minimum_unified(uint2 a, uint2 b)
{
	return min(a, b);
}
template<>
inline __device__ __host__ uint3 minimum_unified(uint3 a, uint3 b)
{
	return min(a, b);
}
template<>
inline __device__ __host__ uint4 minimum_unified(uint4 a, uint4 b)
{
	return min(a, b);
}

template<>
inline __device__ __host__ uint maximum_value_format()
{
	return ~0u;
}
template<>
inline __device__ __host__ uint2 maximum_value_format()
{
	return make_uint2(~0u);
}
template<>
inline __device__ __host__ uint3 maximum_value_format()
{
	return make_uint3(~0u);
}
template<>
inline __device__ __host__ uint4 maximum_value_format()
{
	return make_uint4(~0u);
}

template<>
inline __device__ __host__ int minimum_unified(int a, int b)
{
	return min(a, b);
}
template<>
inline __device__ __host__ int2 minimum_unified(int2 a, int2 b)
{
	return min(a, b);
}
template<>
inline __device__ __host__ int3 minimum_unified(int3 a, int3 b)
{
	return min(a, b);
}
template<>
inline __device__ __host__ int4 minimum_unified(int4 a, int4 b)
{
	return min(a, b);
}

template<>
inline __device__ __host__ int maximum_value_format()
{
	return (1u << 31u) - 1u;
}
template<>
inline __device__ __host__ int2 maximum_value_format()
{
	return make_int2((1u << 31u) - 1u);
}
template<>
inline __device__ __host__ int3 maximum_value_format()
{
	return make_int3((1u << 31u) - 1u);
}
template<>
inline __device__ __host__ int4 maximum_value_format()
{
	return make_int4((1u << 31u) - 1u);
}

template<>
inline __device__ __host__ float minimum_unified(float a, float b)
{
	return fminf(a, b);
}
template<>
inline __device__ __host__ float2 minimum_unified(float2 a, float2 b)
{
	return fminf(a, b);
}
template<>
inline __device__ __host__ float3 minimum_unified(float3 a, float3 b)
{
	return fminf(a, b);
}
template<>
inline __device__ __host__ float4 minimum_unified(float4 a, float4 b)
{
	return fminf(a, b);
}

template<>
inline __device__ __host__ float maximum_value_format()
{
	return INFINITY;
}
template<>
inline __device__ __host__ float2 maximum_value_format()
{
	return make_float2(INFINITY);
}
template<>
inline __device__ __host__ float3 maximum_value_format()
{
	return make_float3(INFINITY);
}
template<>
inline __device__ __host__ float4 maximum_value_format()
{
	return make_float4(INFINITY);
}

template <class T>
__global__ void ____minimum_reduce(T* data, uint length, uint stride)
{
	__shared__ T shared[____block_size_cumsum]; uint internal_idx = threadIdx.x;
	uint grid_idx = (threadIdx.x + ____block_size_cumsum * blockIdx.x) * stride;
	shared[internal_idx] = (grid_idx < length) ? data[grid_idx] : T();

	for (uint i = 0u; i < ____block_pow_cumsum; i++)
	{
		internal_idx <<= 1u;
		__syncthreads();
		if (internal_idx < ____block_size_cumsum)
			shared[internal_idx] = minimum_unified<T>(shared[internal_idx], shared[internal_idx + (1u << i)]);
		__syncthreads();
	}
	if (grid_idx < length && threadIdx.x == 0u)
		data[grid_idx] = shared[threadIdx.x];
}
/// <summary>
/// Mutates the provided buffer to find the smallest value therein. Runs in effective O(log n) time.
/// </summary>
/// <typeparam name="T">: Type of numeric object.</typeparam>
/// <param name="data">: GPU buffer</param>
/// <returns></returns>
template <class T>
T find_minimum_val(smart_gpu_buffer<T>& data)
{
	uint stride = 1u;
	while (stride < data.dedicated_len)
	{
		dim3 blocks(ceilf(data.dedicated_len / (float)(____block_size_cumsum * stride)));
		____minimum_reduce<<<blocks, ____block_size_cumsum>>>(data.gpu_buffer_ptr, data.dedicated_len, stride);
		stride <<= ____block_pow_cumsum;
	}
	T result; cudaMemcpy(&result, data.gpu_buffer_ptr, sizeof(T), cudaMemcpyDeviceToHost);
	return result;
}

#endif