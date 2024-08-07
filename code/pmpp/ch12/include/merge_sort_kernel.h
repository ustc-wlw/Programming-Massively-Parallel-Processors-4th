#pragma once

#include <cuda_runtime.h>

__host__ __device__
void merge_sort_sequencial(const int* A, int lenA, const int* B, int lenB, int* C);

__device__
int co_rank(int out_rank, int* A, int lenA, int* B, int lenB);

__device__
void merge_sort_sequencial_circular(int* As, int lenA, int* Bs, int lenB, int* C,
	int As_start, int Bs_start, int tileSize);

__device__
int co_rank_circular(int out_rank, int* As, int lenA, int* Bs, int lenB,
	int As_start, int Bs_start, int tileSize);

__global__
void meger_basic_kernel(int* A, int lenA, int* B, int lenB, int* C);

__global__
void merge_tiled_kernel(int* A, int lenA, int* B, int lenB, int* C, int tileSize);

__global__
void merge_circular_buffer_kernel(int* A, int lenA, int* B, int lenB, int* C, int tileSize);