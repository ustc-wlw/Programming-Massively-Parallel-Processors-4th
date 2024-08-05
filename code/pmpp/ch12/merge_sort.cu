#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

#include "merge_sort_kernel.cuh"

__device__
void merge_sort_sequencial(int* A, int mLen, int* B, int nLen, int* C) {
	int len = mLen <= nLen ? mLen : nLen;
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
	int indexA = out_rank > lenA ? lenA : out_rank;
	int indexB = out_rank - indexA;
	int indexA_low = out_rank > lenB ? out_rank - lenB : 0;
	int indexB_low = out_rank > lenA ? out_rank - lenA : 0;
	int delta;
	bool activate = true;

	while (activate)
	{
		if (indexA >= indexA_low && indexB <= lenB && A[indexA - 1] > B[indexB]) {
			delta = (indexA - indexA_low + 1) >> 1;
			indexA -= delta;
			indexB_low = indexB;
			indexB += delta;// indexA + indexB = out_rank
		}
		else if (indexA < lenA && indexB >= indexB_low && A[indexA] <= B[indexB - 1]) {
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
	int elementsPerThread = (lenA + lenB + blockDim.x * gridDim.x - 1) / blockDim.x * gridDim.x;

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
void merge_tiled_kernel(int* A, int lenA, int* B, int lenB, int* C, int tileSize) {

	// shared memory size is 2 * tileSize
	// each tileSize size for A and B sub array
	extern __shared__ int sharedAB[];
	int *As = sharedAB;
	int *Bs = sharedAB + tileSize;

	// start point of array C for this block
	int c_start_for_block = blockIdx.x * ceil((lenA + lenB)*1.0 / gridDim.x);
	// end point of array C for this block
	int c_end_for_block = (blockIdx.x + 1) * ceil((lenA + lenB) * 1.0 / gridDim.x);
	c_end_for_block = min(c_end_for_block, (lenA + lenB));

	if (threadIdx.x == 0)
	{
		// get start and end offset of array A to generate C[c_start_for_block ~ c_end_for_block]
		// store in shared memory to make visuable for all threads in this block
		As[0] = co_rank(c_start_for_block, A, lenA, B, lenB);
		As[1] = co_rank(c_end_for_block, A, lenA, B, lenB);
	}
	__syncthreads();

	int A_start_offset = As[0];
	int A_end_offset = As[1];
	// start and end offset of array B to generate C[c_start_for_block ~ c_end_for_block]
	int B_start_offset = c_start_for_block - A_start_offset;
	int B_end_offset = c_end_for_block - A_end_offset;

	__syncthreads();

	int counter = 0;
	int C_len = c_end_for_block - c_start_for_block;
	int totalCount = ceil(C_len / tileSize);
	int A_len = A_end_offset - A_start_offset;
	int B_len = B_end_offset - B_start_offset;

	// all blocks start their tiles from A_curr or B_curr
	int* A_curr = A + A_start_offset;
	int* B_curr = B + B_start_offset;
	int* C_curr = C + c_start_for_block;

	// A_consumed and B_consumed give the total number
	// of A and B elements consumed by the thread block in previous iterations of the
	//	while - loop
	int A_consumed_total = 0;
	int B_consumed_total = 0;
	int C_completed = 0;

	// load sub array of A and B in HBM to shared memory totalCount times
	// each time load tileSize number elements
	while (counter < totalCount)
	{
		// iterate tileSize / blockDim.x times
		// each time load blockDim.x elements for every block

		/*
		 For each iteration of the while-loop, the starting point for loading the current tile in the A and B array
		 depends on the total number of A and B elements that have been consumed by all
         threads of the block during the previous iterations of the while-loop
		*/
		for (size_t i = 0; i < tileSize; i+= blockDim.x)
		{
			if (i + threadIdx.x < A_len - A_consumed_total) {
				As[i + threadIdx.x] = A_curr[A_consumed_total + i + threadIdx.x];
			}
		}
		for (size_t i = 0; i < tileSize; i += blockDim.x)
		{
			if (i + threadIdx.x < B_len - B_consumed_total) {
				Bs[i + threadIdx.x] = B_curr[B_consumed_total + i + threadIdx.x];
			}
		}

		__syncthreads();

		// after tileSize elements have been load into shared memory
		// assigning a section of the output to each thread and running the co - rank function to determine
		// the sections of shared memory data that should be used for generating that output section
		int section_size = ceil(tileSize * 1.0 / blockDim.x);
		int out_start_offset_for_thread = threadIdx.x * section_size;
		int out_end_offset_for_thred = (threadIdx.x + 1) * section_size;
		// check output index is not out of range
		out_start_offset_for_thread = out_start_offset_for_thread <= (C_len - C_completed)
			? out_start_offset_for_thread : C_len - C_completed;
		out_end_offset_for_thred = out_end_offset_for_thred <= (C_len - C_completed)
			? out_end_offset_for_thred : C_len - C_completed;

		// in last iteration, valid elements of array A and B may less than tileSize
		int As_start_offset_for_thread = co_rank(out_start_offset_for_thread, As, min(tileSize, A_len - A_consumed_total),
												Bs, min(tileSize, B_consumed_total));
		int As_end_offset_for_thread = co_rank(out_end_offset_for_thred, As, min(tileSize, A_len - A_consumed_total),
												Bs, min(tileSize, B_consumed_total));
		int A_consumed = As_end_offset_for_thread - As_start_offset_for_thread;

		int Bs_start_offset_for_thread = out_start_offset_for_thread - As_start_offset_for_thread;
		int Bs_end_offset_for_thread = out_end_offset_for_thred - As_end_offset_for_thread;
		int B_consumed = Bs_end_offset_for_thread - Bs_start_offset_for_thread;

		merge_sort_sequencial(As + As_start_offset_for_thread, A_consumed,
			Bs + Bs_start_offset_for_thread, B_consumed,
			C_curr + C_completed + out_start_offset_for_thread);

		counter++;

		// Collectively, the total number of A elements and B elements that are used by
		// all threads in a block for the iteration will add up to tileSize
		C_completed += tileSize;
		A_consumed_total += co_rank(tileSize, As, tileSize, Bs, tileSize);
		B_consumed_total += C_completed - A_consumed_total;
		

		__syncthreads();
	}
}


__global__
void merge_circular_buffer_kernel(int* A, int lenA, int* B, int lenB, int* C, int tileSize) {

}