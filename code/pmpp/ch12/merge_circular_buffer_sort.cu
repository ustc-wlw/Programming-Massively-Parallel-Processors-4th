#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "merge_sort_kernel.h"

static constexpr int TILE_SIZE=256;

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
	if (out_rank == 0)
	{
		return 0;
	}
	
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
		if (i > 0 && j < lenB && As[i_minus1_cir] > Bs[j_cir]) {
			delta = (i - i_low + 1) >> 1;
			i -= delta;
			j_low = j;
			j += delta;// i + j = out_rank
		}
		else if (i < lenA && j > 0 && As[i_cir] <= Bs[j_minus1_cir]) {
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
	// extern __shared__ int ABs[];
	__shared__ int ABs[2 * TILE_SIZE];
	int* As = ABs;
	int* Bs = ABs + tileSize;

	int C_block_start = blockIdx.x * ceil(float(lenA + lenB) / gridDim.x);
	int C_block_end = (blockIdx.x + 1) * ceil(float(lenA + lenB) / gridDim.x);
	C_block_start = min(C_block_start, lenA + lenB);
	C_block_end = min(C_block_end, lenA + lenB);

	if (threadIdx.x == 0)
	{
		As[0] = co_rank_circular(C_block_start, A, lenA, B, lenB, 0, 0, 1);
		As[1] = co_rank_circular(C_block_end, A, lenA, B, lenB, 0, 0, 1);
	}

	__syncthreads();

	// start and end offset of array A to generate C[C_block_start ~ C_block_end]
	int A_start_offset = As[0];
	int A_end_offset = As[1];
	if (threadIdx.x == 0)
	{
		printf("block id %d, C_block_start: %d, C_block_end: %d\n", blockIdx.x, C_block_start, C_block_end);
		printf("block id %d, A_start_offset: %d, A_end_offset: %d\n",blockIdx.x, A_start_offset, A_end_offset);
	}
	
	// start and end offset of array B to generate C[C_block_start ~ C_block_end]
	int B_start_offset = C_block_start - A_start_offset;
	int B_end_offset = C_block_end - A_end_offset;

	__syncthreads();

	int counter = 0;
	// the number of elements of C this thread block generate
	int C_len = C_block_end - C_block_start;
	// at each iteration generate tileSize elements, so loop iterationNumber times
	int iterationNumber = ceil(float(C_len) / tileSize);
	int A_len = A_end_offset - A_start_offset;
	int B_len = B_end_offset - B_start_offset;

	// A_loaded_total and B_loaded_total represent the total number
	// of elements ranged in A[A_start_offset : A_end_offset] 
	// and B[B_start_offset : B_end_offset] repsectively loaded to shared memory 
	// by the thread block in previous iterations of the while - loop
	int A_loaded_total = 0;
	int B_loaded_total = 0;
	int A_consumed = 0;
	int B_consumed = 0;
	int C_generated = 0; // C elements already produced by previous iterations

	// start offset of circular shared memory buffer
	int As_start = 0;
	int Bs_start = 0;
	int As_consumed = tileSize;// consumed As elements for current iteration
	int Bs_consumed = tileSize;

	while (counter < iterationNumber)
	{
		// load As_consumed elments of A at single iteration
		for (size_t i = threadIdx.x; i < As_consumed; i+= blockDim.x)
		{
			if (i < lenA - A_loaded_total) {
				// new loaded element start offset in As is (As_start + tileSize - As_consumed)
				As[(As_start + tileSize - As_consumed + i) % tileSize] = A[A_start_offset + A_loaded_total + i];
			}
		}
		// load Bs_consumed elments of B at single iteration
		for (size_t i = threadIdx.x; i < Bs_consumed; i+= blockDim.x)
		{
			if (i < lenB - B_loaded_total) {
				Bs[(Bs_start + tileSize - Bs_consumed  + i) % tileSize] = B[B_start_offset + B_loaded_total + i];
			}
		}

		__syncthreads();

		// at single iteration, each block generate tileSize elements of array C(except last iteration if remaining C elements is less than tileSize)
		// so each thread generate ceil(tileSize / threadBlock size) elements
		// caculate output elements indexes by thread id
		int k_cur = threadIdx.x * ceil(float(tileSize) / blockDim.x);
		int k_next = (threadIdx.x + 1) * ceil(float(tileSize) / blockDim.x);
		// ensure output index is in range of C_block_end
		k_cur = min(k_cur, C_len - C_generated);
		k_next = min(k_next, C_len - C_generated);

		// start index in As to generate current sub array starting with k_cur for this thread
		int a_start_cur = co_rank_circular(k_cur, As, min(tileSize, A_len - A_consumed),
										Bs, min(tileSize, A_len - B_consumed),
										As_start, Bs_start, tileSize);
		// start index in Bs to generate sub array starting with k_cur for this thread
		int b_start_cur = k_cur - a_start_cur;
		// start index in As to generate next sub array starting with k_next
		int a_start_next = co_rank_circular(k_next, As, min(tileSize, A_len - A_consumed),
										Bs, min(tileSize, B_len - B_consumed),
										As_start, Bs_start, tileSize);
		// start index in Bs to generate next sub array starting with k_next
		int b_start_next = k_next - a_start_next;

		if (threadIdx.x == 0)
		{
			printf("block id %d, k_cur: %d, k_next: %d\n", blockIdx.x, k_cur, k_next);
			printf("block id %d, a_start_cur: %d, a_start_next: %d\n",blockIdx.x, a_start_cur, a_start_next);
		}
		// all thread call merge_sort_sequencial_circular with array index above
		merge_sort_sequencial_circular(As, a_start_next - a_start_cur,
									Bs, b_start_next - b_start_cur,
									C + C_block_start + C_generated + k_cur,
									As_start  + a_start_cur, Bs_start + b_start_cur, tileSize);

		// update block level variabel that are visiable to all threads in block
		// like shared memory elments consumed by this thread block at current iteration
		// in case last iteration that remaining elements less than tileSize
		As_consumed = co_rank_circular(min(tileSize, C_len - C_generated),
										As, min(tileSize, A_len - A_consumed),
										Bs, min(tileSize, B_len - B_consumed),
										As_start, Bs_start, tileSize);
		Bs_consumed = min(tileSize, C_len - C_generated) - As_consumed;
		A_consumed += As_consumed;
		B_consumed += Bs_consumed;
		if (threadIdx.x == 0)
		{
			printf("block id %d, As_consumed: %d, Bs_consumed: %d\n", blockIdx.x, As_consumed, Bs_consumed);
		}
		
		// update shared memory start offset
		As_start = (As_start + As_consumed) % tileSize;
		Bs_start = (Bs_start + Bs_consumed) % tileSize;

		C_generated += min(tileSize, C_len - C_generated);
		A_loaded_total += min(tileSize, C_len - C_generated);
		B_loaded_total += min(tileSize, C_len - C_generated);

		counter++;

		__syncthreads();
	}
}