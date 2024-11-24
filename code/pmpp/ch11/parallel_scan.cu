#include <stdio.h>
#include <cuda_runtime.h>

#include "parallel_scan.h"

static constexpr int COARSE_FACTOR = 4;


// computional complexity is O(N)
__host__ __device__
void sequential_scan(float* input, int len, float* out) {
	if (input && len > 0)
	{
		out[0] = input[0];
		for (size_t i = 1; i < len; i++)
		{
			out[i] = out[i - 1] + input[i];
		}
	}
}


// inclusive scan(add operation)
// computional complexity is O(N * logN) N is input length
__global__
void kogge_stone_scan_kernel(float* input, int len, float* out) {
	// use shared memory
	__shared__ float shm[SECTION_SIZE];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len)
	{
		shm[threadIdx.x] = input[idx];
	}
	else {
		shm[threadIdx.x] = 0;
	}

	// reduction tree caculate
	for (size_t stride=1; stride < SECTION_SIZE; stride *= 2)
	{
		__syncthreads();

		// store added value in tmp variable to avoid write-after-read race condition between threads
		float tmp;
		if (threadIdx.x >= stride) {
			tmp = shm[threadIdx.x] + shm[threadIdx.x - stride];
		}
		__syncthreads();

		if (threadIdx.x >= stride) {
			shm[threadIdx.x] = tmp;
		}
	}

	if (idx < len)
	{
		out[idx] = shm[threadIdx.x];
	}
}


// with two shared memory buffers to avoid __syncthreads
__global__
void kogge_stone_scan_double_buffer_kernel(float* input, int len, float* out) {
	// use shared memory
	__shared__ float shm1[SECTION_SIZE];
	__shared__ float shm2[SECTION_SIZE];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < len)
	{
		shm1[threadIdx.x] = input[idx];
	}
	else {
		shm1[threadIdx.x] = 0;
	}

	int flag = 1;

	for (size_t stride = 1; stride < SECTION_SIZE; stride *= 2)
	{
		__syncthreads();

		if (threadIdx.x >= stride) {
			if (flag % 2 != 0)
			{
				// read from shm1 and write to shm2
				shm2[threadIdx.x] = shm1[threadIdx.x] + shm1[threadIdx.x - stride];
			}
			else {
				// read from shm2 and write to shm1
				shm1[threadIdx.x] = shm2[threadIdx.x] + shm2[threadIdx.x - stride];
			}
		}
		flag++;
	}

	if (idx < len)
	{
		if (flag % 2 == 0)
		{
			out[idx] = shm2[threadIdx.x];
		}
		else {
			out[idx] = shm1[threadIdx.x];
		}
	}
}


// exclusive scan(add operation)
__global__
void kogge_stone_exclusive_scan_kernel(float* input, int len, float* out) {
	// use shared memory
	__shared__ float shm[SECTION_SIZE];

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	// can only change this four lines based on kogge_stone_scan_kernel
	if (idx < len && idx != 0)
	{
		shm[threadIdx.x] = input[idx - 1];
	}
	else {
		shm[threadIdx.x] = 0;
	}

	for (size_t stride = 1; stride < SECTION_SIZE; stride *= 2)
	{
		__syncthreads();

		// store added value in tmp variable to avoid write-after-read race condition between threads
		float tmp;
		if (threadIdx.x >= stride) {
			tmp = shm[threadIdx.x] + shm[threadIdx.x - stride];
		}
		__syncthreads();

		if (threadIdx.x >= stride) {
			shm[threadIdx.x] = tmp;
		}
	}

	if (idx < len)
	{
		out[idx] = shm[threadIdx.x];
	}
}


// inclusive scan based on Brent Kung algorithm
__global__
void brent_kung_scan_kernel(float* input, int len, float* out) {
	// use shared memory
	__shared__ float shm[SECTION_SIZE];// assert SECTION_SIZE == blockDim.x

	// each thread block caculate 2 * SECTION_SIZE elements
	int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
	if (idx < len)
	{
		shm[threadIdx.x] = input[idx];
	}
	if (idx + blockDim.x < len) 
	{
		shm[threadIdx.x + blockDim.x] = input[idx + blockDim.x];
	}
	
	// reduction tree caculation
	for (size_t stride = 1; stride < len; stride *= 2)
	{
		__syncthreads();
		// complex thread index mapping to data
		// to decrease control diverage
		int dataIndex = (threadIdx.x + 1) * 2 * stride - 1;
		if (dataIndex < len)
		{
			shm[dataIndex] += shm[dataIndex - stride];
		}
	}

	// reverse tree to distribute intermedia acculation results of reduction tree
	// to other output elements
	for (size_t stride = SECTION_SIZE / 4; stride >= 1; stride /= 2)
	{
		__syncthreads();
		// complex thread index mapping to data
		// to decrease control divergence
		int dataIndex = (threadIdx.x + 1) * 2 * stride - 1;
		if (dataIndex + stride < len)
		{
			shm[dataIndex + stride] += shm[dataIndex];
		}
	}

	__syncthreads();

	if (idx < len)
	{
		out[idx] = shm[threadIdx.x];
	}
	if (idx + blockDim.x < len)
	{
		out[idx + blockDim.x] = shm[threadIdx.x + blockDim.x];
	}
}


// thread coarsening with COARSE_FACTOR
__global__
void kogge_stone_scan_thread_coarse_kernel(float* input, int len, float* out) {
	// use shared memory
	__shared__ float shm[COARSE_FACTOR * SECTION_SIZE];// assert SECTOR_SIZE = block size
	int block_start = blockIdx.x * blockDim.x * COARSE_FACTOR;
	for (size_t i = threadIdx.x; i < len - block_start; i+= blockDim.x)
	{
		shm[i] = input[block_start + i];
	}

	// phase 1: sequential_scan on shared memory
	// each thread scan continuous COARSE_FACTOR elements
	sequential_scan(shm + threadIdx.x * COARSE_FACTOR, COARSE_FACTOR, shm + threadIdx.x * COARSE_FACTOR);

	__syncthreads();

	// phase 2: kogge_stone_scan on logical array whose each elements is the last one
	// of the subarray owned by each thread after phase 1
	int index = (threadIdx.x + 1) * COARSE_FACTOR - 1;

	// reduction tree caculate
	for (size_t stride = COARSE_FACTOR; stride < COARSE_FACTOR * SECTION_SIZE; stride *= 2)
	{
		// store added value in tmp variable to avoid write-after-read race condition between threads
		float tmp;
		if (index >= stride) {
			tmp = shm[index] + shm[index - stride];
		}
		__syncthreads();

		if (index >= stride) {
			shm[index] = tmp;
		}

		__syncthreads();
	}

	// phase 3: each thread adds the new value of the last element of its predecessor��s section
	// to its elements.The last elements of each subsection do not need to
	// be updated during this phase.
	if (threadIdx.x >= 1)
	{	// ignore last element
		for (size_t i = 0; i < COARSE_FACTOR - 1; i++)
		{
			shm[threadIdx.x * COARSE_FACTOR + i] = shm[(threadIdx.x - 1) * COARSE_FACTOR + i];
		}
	}

	// update output data
	for (size_t i = threadIdx.x; i < len - block_start; i += blockDim.x)
	{
		out[block_start + i] = shm[i];
	}
}


