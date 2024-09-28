#ifndef CUM_SUM
#define CUM_SUM

#include "CUDA_memory.h"
#include "helper_math.h"

__device__ constexpr uint block_pow_cumsum = 9u;
__device__ constexpr uint block_size_cumsum = 1u << block_pow_cumsum;

template <class T>
__global__ void cum_sum_reduce(T* data, uint length, uint stride)
{
	__shared__ T shared[block_size_cumsum]; uint internal_idx = threadIdx.x;
	uint grid_idx = (threadIdx.x + block_size_cumsum * blockIdx.x + 1u) * stride - 1u;
	shared[internal_idx] = (grid_idx < length) ? data[grid_idx] : T();

	for (uint i = 0u; i < block_pow_cumsum; i++)
	{
		internal_idx = (internal_idx << 1u) + 1u;
		__syncthreads();
		if (internal_idx < block_size_cumsum)
			shared[internal_idx] += shared[internal_idx - (1u << i)];
		__syncthreads();
	}
	if (grid_idx < length)
		data[grid_idx] = shared[threadIdx.x];
}

template <class T>
__global__ void cum_sum_post(T* data, uint length, uint stride)
{
	__shared__ T shared[block_size_cumsum];
	uint internal_idx = ((threadIdx.x + 1u) << (block_pow_cumsum - 1u)) - 1u;
	uint grid_idx = (threadIdx.x + block_size_cumsum * blockIdx.x + 1u) * stride - 1u;
	shared[threadIdx.x] = (grid_idx < length) ? data[grid_idx] : T();

	bool add = false; // propagate across blocks
	for (uint i = 0; i < block_pow_cumsum; i++)
		add |= (threadIdx.x + 1u) == (1 << i);
	if (blockIdx.x > 0 && add)
		shared[threadIdx.x] += data[block_size_cumsum * blockIdx.x * stride - 1u];

	for (int i = block_pow_cumsum - 2; i >= 0; i--)
	{
		__syncthreads();
		if ((internal_idx + (1u << i)) < block_size_cumsum)
			shared[internal_idx + (1u << i)] += shared[internal_idx];
		__syncthreads();

		internal_idx >>= 1u;
	}
	if (grid_idx < length)
		data[grid_idx] = shared[threadIdx.x];
}

/// <summary>
/// Applies an inclusive cumulative sum to a GPU buffer of any relevant type. Runs in O(n log n) time.
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
		dim3 blocks(ceilf(data.dedicated_len / (float)(block_size_cumsum * stride)));
		cum_sum_reduce<<<blocks, block_size_cumsum>>>(data.gpu_buffer_ptr, data.dedicated_len, stride);
		stride <<= block_pow_cumsum; cuda_sync();
	}
	stride >>= block_pow_cumsum;
	while (stride > 0)
	{
		dim3 blocks(ceilf(data.dedicated_len / (float)(block_size_cumsum * stride)));
		cum_sum_post<<<blocks, block_size_cumsum>>>(data.gpu_buffer_ptr, data.dedicated_len, stride);
		stride >>= block_pow_cumsum; cuda_sync();
	}
}

#endif