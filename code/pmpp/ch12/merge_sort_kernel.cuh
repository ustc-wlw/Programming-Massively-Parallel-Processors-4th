#pragma

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>

__device__
void merge_sort_sequencial(int* A, int mLen, int* B, int nLen, int* C);

__device__
int co_rank(int out_rank, int* A, int lenA, int* B, int lenB);

__device__
void merge_sort_sequencial_circular(int* As, int lenA, int* Bs, int lenB, int* C,
	int As_start, int Bs_start, int tileSize);

__device__
int co_rank_circular(int out_rank, int* As, int lenA, int* Bs, int lenB,
	int As_start, int Bs_start, int tileSize);