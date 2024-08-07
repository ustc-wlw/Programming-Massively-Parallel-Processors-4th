#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "merge_sort_kernel.h"

static constexpr int TILE_SIZE=256;

__host__ __device__
void merge_sort_sequencial(const int* A, int mLen, const int* B, int nLen, int* C) {
	int A_index = 0;
	int B_index = 0;
	int k = 0;
	while (A_index < mLen && B_index < nLen)
	{
		if (A[A_index] <= B[B_index]) {
			C[k++] = A[A_index];
			A_index++;
		}
		else {
			C[k++] = B[B_index];
			B_index++;
		}
	}

	if (A_index < mLen)
	{
		for (int i = A_index; i < mLen; i++) {
			C[k++] = A[i];
		}
	}

	if (B_index < nLen)
	{
		for (int i = B_index; i < nLen; i++) {
			C[k++] = B[i];
		}
	}

}


__device__
int co_rank(int out_rank, int* A, int lenA, int* B, int lenB) {
	if (out_rank == 0 || lenA == 0)
	{
		return 0;
	}
	
	int indexA = out_rank > lenA ? lenA : out_rank;
	int indexB = out_rank - indexA;
	int indexA_low = out_rank > lenB ? out_rank - lenB : 0;
	int indexB_low = out_rank > lenA ? out_rank - lenA : 0;
	int delta;
	bool activate = true;

	while (activate)
	{
		if (indexA > 0 && indexB < lenB && A[indexA - 1] > B[indexB]) {
			delta = (indexA - indexA_low + 1) >> 1;
			indexA -= delta;
			indexB_low = indexB;
			indexB += delta;// indexA + indexB = out_rank
		}
		else if (indexA < lenA && indexB > 0 && A[indexA] <= B[indexB - 1]) {
			delta = (indexB - indexB_low + 1) >> 1;
			indexB -= delta;
			indexA_low = indexA;
			indexA += delta;// indexA + indexB = out_rank
		}
		else {
			activate = false;
		}
	}

	return indexA;
}

__global__
void meger_basic_kernel(int* A, int lenA, int* B, int lenB, int* C) {
	// output C length is lenA + lenB
	
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int elementsPerThread = ceil(float(lenA + lenB)) / blockDim.x * gridDim.x;

	int start_offset = idx * elementsPerThread;
	int end_offset = min((idx + 1) * elementsPerThread, lenA + lenB);

	int A_start_offset = co_rank(start_offset, A, lenA, B, lenB);
	int B_start_offset = start_offset - A_start_offset;

	int A_end_offset = co_rank(end_offset, A, lenA, B, lenB);
	int B_end_offset = end_offset - A_end_offset;

	merge_sort_sequencial(A + A_start_offset, A_end_offset - A_start_offset,
		B + B_start_offset, B_end_offset - B_start_offset,
		C + start_offset);
}


__global__
void merge_tiled_kernel(int* A, int lenA, int* B, int lenB, int* C, int tileSize=TILE_SIZE) {

	// shared memory size is 2 * tileSize
	// each tileSize size for A and B sub array
	// extern __shared__ int sharedAB[];
	__shared__ int sharedAB[2 * TILE_SIZE];
	int *As = sharedAB;
	int *Bs = sharedAB + tileSize;

	// start offset of array C for this block
	int C_block_start = blockIdx.x * ceil(float(lenA + lenB) / gridDim.x);
	// end offset of array C for this block
	int C_block_end = (blockIdx.x + 1) * ceil(float(lenA + lenB) / gridDim.x);
	C_block_end = min(C_block_end, (lenA + lenB));
	
	if (threadIdx.x == 0)
	{
		// get start and end offset of array A to generate C[C_block_start ~ C_block_end]
		// store them in shared memory to make visuable for all threads in this block
		As[0] = co_rank(C_block_start, A, lenA, B, lenB);
		As[1] = co_rank(C_block_end, A, lenA, B, lenB);
	}
	__syncthreads();

	int A_start_offset = As[0];
	int A_end_offset = As[1];
	// start and end offset of array B to generate C[C_block_start ~ C_block_end]
	int B_start_offset = C_block_start - A_start_offset;
	int B_end_offset = C_block_end - A_end_offset;

	__syncthreads();

	int counter = 0;
	int C_len = C_block_end - C_block_start;
	int totalCount = ceil(float(C_len) / tileSize);
	int A_len = A_end_offset - A_start_offset;
	int B_len = B_end_offset - B_start_offset;

	// A_consumed and B_consumed represent the total number of elements of
	// A and B consumed by this thread block in previous iterations of the
	//	while - loop
	int A_consumed_total = 0;
	int B_consumed_total = 0;
	int C_completed = 0;

	// load sub array of A and B in HBM to shared memory totalCount times
	// each time load tileSize number elements
	// at each iteration of the while-loop, the starting point of the loaded tile in array A and B
	// depends on the A_consumed_total and B_consumed_total
	while (counter < totalCount)
	{
		// iterate tileSize / blockDim.x times
		// each time load blockDim.x elements for every block
		for (size_t i = threadIdx.x; i < tileSize; i+= blockDim.x)
		{
			if (i < A_len - A_consumed_total) {
				As[i] = A[A_start_offset + A_consumed_total + i];
			}
		}
		for (size_t i = threadIdx.x; i < tileSize; i += blockDim.x)
		{
			if (i < B_len - B_consumed_total) {
				Bs[i] = B[B_start_offset + B_consumed_total + i];
			}
		}
		__syncthreads();

		// after tileSize elements have been load into shared memory
		// assigning a section of the output to each thread and running the co - rank function to locate
		// the according part of shared memory data that should be used for generating that output section
		int section_size = ceil(tileSize * 1.0 / blockDim.x);
		int c_curr = threadIdx.x * section_size;
		int c_next = (threadIdx.x + 1) * section_size;
		// check output index is not out of range
		c_curr = min(c_curr, C_len - C_completed);
		c_next = min(c_next, C_len - C_completed);

		// in last iteration, remaining elements of sub array of A and B may less than tileSize
		int As_curr = co_rank(c_curr, As, min(tileSize, A_len - A_consumed_total),
												Bs, min(tileSize, B_len - B_consumed_total));
		int As_next = co_rank(c_next, As, min(tileSize, A_len - A_consumed_total),
												Bs, min(tileSize,  B_len - B_consumed_total));
		int A_consumed = As_next - As_curr;

		int Bs_curr = c_curr - As_curr;
		int Bs_next = c_next - As_next;
		int B_consumed = Bs_next - Bs_curr;
		
		merge_sort_sequencial(As + As_curr, A_consumed,
			Bs + Bs_curr, B_consumed,
			C + C_block_start + C_completed + c_curr);

		// Collectively, the total number of A elements and B elements that are used by
		// all threads in a block for the iteration will add up to tileSize
		A_consumed_total +=  co_rank(min(tileSize, C_len - C_completed),
									As, min(tileSize, A_len - A_consumed_total),
									Bs, min(tileSize, B_len - B_consumed_total));
		C_completed += min(tileSize, C_len - C_completed);
		B_consumed_total += C_completed - A_consumed_total;
		
		counter++;

		__syncthreads();
		
	}
}
