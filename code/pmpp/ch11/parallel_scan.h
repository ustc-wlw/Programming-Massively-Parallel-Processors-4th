#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

constexpr int SECTION_SIZE = 32;
constexpr int BLOCK_SIZE = SECTION_SIZE;

__host__ __device__
void sequential_scan(float* input, int len, float* out);

__global__
void kogge_stone_scan_kernel(float* input, int len, float* out);

__global__
void brent_kung_scan_kernel(float* input, int len, float* out);

__global__
void kogge_stone_scan_thread_coarse_kernel(float* input, int len, float* out);

