#include "parallel_scan.h"


// inclusive scan(add operation)
// param "blocks_scan" store the last element of segment scaned result of this block
__global__
void segmented_kogge_stone_scan_kernel(float* input, int len, float* out, float *blocks_scan) {
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

	// store this block local accumulated scan to global memory
	__syncthreads();

	if (threadIdx.x == blockDim.x - 1)
	{
		blocks_scan[blockIdx.x] = shm[threadIdx.x];
	}
}

__global__
void distribute_blocks_scan_kernel(float* S, int len, float* out) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (blockIdx.x >= 1 && idx < len)
	{
		out[idx] += S[blockIdx.x - 1];
	}
}

void three_phases_segmented_scan(float *input_h, int len, float *output_h) {
	int size_input = sizeof(float) * len;
	int size_output = size_input;

	int S_len = ceil(float(size_input) / SECTION_SIZE);
	int size_S = sizeof(float) * S_len;

	float* input_d, * output_d;
	cudaMalloc((void**)&input_d, size_input);
	cudaMalloc((void**)&output_d, size_output);
	cudaMemset(output_d, 0, size_output);

	cudaStream_t stream;
	cudaStreamCreate(&stream);

	cudaMemcpyAsync(input_d, input_h, size_input, cudaMemcpyHostToDevice, stream);

	float* S;
	cudaMalloc((void**)&S, size_S);
	cudaMemset(S, 0, size_S);

	// stage 1: call segmented_kogge_stone_scan_kernel
	dim3 block(SECTION_SIZE);
	dim3 grid(ceil(float(size_input) / SECTION_SIZE));
	printf("Launch segmented_kogge_stone_scan_kernel with grid: %d, block: %d\n", grid.x, block.x);
	segmented_kogge_stone_scan_kernel <<<grid, block, 0, stream >>> (input_d, len, output_d, S);

	// stage 2: scan on S which store local accumulated sum of each thread block
	// to get glbal accumulated sum of each block
	sequential_scan(S, S_len, S);

	// stage 3: the threads in a block add the sum of all previous scan blocks
	// to the elements of their scan block
	distribute_blocks_scan_kernel << <gird, block, 0, stream >> > (S, len, output_d);

	cudaMemcpyAsync(output_h, output_d, size_output, cudaMemcpyDeviceToHost, stream);

	cudaStreamSynchronize(stream);

	cudaFree(input_d);
	cudaFree(output_d);
	cudaFree(S);

	cudaStreamDestroy(stream);
}