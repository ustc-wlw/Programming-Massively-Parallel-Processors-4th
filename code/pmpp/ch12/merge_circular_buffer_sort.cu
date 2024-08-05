#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "merge_sort_kernel.cuh"

// use of circular buffers to allow us to make full use of the data
// loaded from global memory
// As and Bs are circular shared memory buffers
__device__
void merge_sort_sequencial_circular(int* As, int lenA, int* Bs, int lenB, int* C,
									int As_start, int Bs_start, int tileSize) {
	int i = 0;
	int j = 0;
	int k = 0;
	while (i < lenA && j < lenB)
	{
		int As_cir = (As_start + i) % tileSize;
		int Bs_cir = (Bs_start + j) % tileSize;
		if (As[As_cir] <= Bs[Bs_cir]) {
			C[k++] = As[As_cir];
			i++;
		}
		else {
			C[k++] = Bs[Bs_cir];
			j++;
		}
	}

	if (i < lenA)
	{
		while (i < lenA) {
			C[k++] = As[(As_start + i) % tileSize];
			i++;
		}
	}

	if (j < lenB)
	{
		while (j < lenB) {
			C[k++] = Bs[(Bs_start + j) % tileSize];
			j++;
		}
	}
}


// use of circular buffers to allow us to make full use of the data
// loaded from global memory
// As and Bs are circular shared memory buffers
__device__
int co_rank_circular(int out_rank, int* As, int lenA, int* Bs, int lenB,
			int As_start, int Bs_start, int tileSize) {
	int i = out_rank > lenA ? lenA : out_rank;
	int j = out_rank - i;
	int i_low = out_rank > lenB ? out_rank - lenB : 0;
	int j_low = out_rank > lenA ? out_rank - lenA : 0;
	int delta;
	bool activate = true;

	// binary search to find start offset of As 
	// that this subarray generate output array starting offset is out_rank
	while (activate)
	{
		// the mapped real index in circular shared memory buffer by current index i
		int i_cir = (As_start + i) % tileSize;
		int i_minus1_cir = (As_start + i - 1) % tileSize;
		int j_cir = (Bs_start + j) % tileSize;
		int j_minus1_cir = (Bs_start + j - 1) % tileSize;
		if (i >= i_low && j <= lenB && As[i_minus1_cir] > Bs[j_cir]) {
			delta = (i - i_low + 1) >> 1;
			i -= delta;
			j_low = j;
			j += delta;// i + j = out_rank
		}
		else if (i < lenA && j >= j_low && As[i_cir] <= Bs[j_minus1_cir]) {
			delta = (j - j_low + 1) >> 1;
			j -= delta;
			i_low = i;
			i += delta;// i + j = out_rank
		}
		else {
			activate = false;
		}
	}
	return i;
}


__global__
void merge_circular_buffer_kernel(int* A, int lenA, int* B, int lenB, int* C, int tileSize) {
	// circular shared memory buffer, size is 2 * tileSize
	extern __shared__ int ABs[];
	int* As = ABs;
	int* Bs = ABs + tileSize;

	int C_block_start = blockIdx.x * ceil((lenA + lenB) * 1.0 / gridDim.x);
	int C_block_end = (blockIdx.x + 1) * ceil((lenA + lenB) * 1.0 / gridDim.x);
	C_block_end = min(C_block_end, lenA + lenB);

	if (threadIdx.x == 0)
	{
		As[0] = co_rank(C_block_start, A, lenA, B, lenB);
		As[1] = co_rank(C_block_end, A, lenA, B, lenB);
	}

	__syncthreads();

	int A_start_offset = As[0];
	int A_end_offset = As[1];
	// start and end offset of array B to generate C[c_start_for_block ~ c_end_for_block]
	int B_start_offset = C_block_start - A_start_offset;
	int B_end_offset = C_block_end - A_end_offset;

	__syncthreads();

	int counter = 0;
	// the number of elements of C this thread block generate
	int C_len = C_block_end - C_block_start;
	// at each iteration generate tileSize elements, so loop iterationNumber times
	int iterationNumber = ceil(C_len / tileSize);
	int A_len = A_end_offset - A_start_offset;
	int B_len = B_end_offset - B_start_offset;

	// A_consumed_total and B_consumed_total give the total number
	// of A[A_start_offset : A_end_offset] 
	// and B[B_start_offset : B_end_offset] elements consumed by the thread block
	// in previous iterations of the while - loop
	int A_consumed_total = 0;
	int B_consumed_total = 0;
	int C_generated = 0;

	// start offset of circular shared memory buffer
	int As_start = 0;
	int Bs_start = 0;
	int As_consumed = tileSize;// consumed As elements for current iteration
	int Bs_consumed = tileSize;

	while (counter < iterationNumber)
	{
		for (size_t i = threadIdx.x; i < tileSize; i+= blockDim.x)
		{
			if (i < lenA - A_consumed_total && i < As_consumed) {

			}
		}

		counter++;

		__syncthreads();
	}
}