#include "parallel_scan.h"


// inclusive scan(add operation)
// stream-based or domino-styl scan algorithm
// partial sum data is passed in one direction
// through the global memory between adjacent thread blocks in the same grid

// flags: used to indicate whether current block scan finished(0 is no, 1 is yes)
// blocks_scan: store block local partial scan sum in global memory
__global__
void segmented_kogge_stone_scan_kernel(float* input, int len, float* out,
									float *flags, float* blocks_scan) {
	__shared__ int bid_s;//block id
	if (threadIdx.x == 0)
	{
		// generate sequential block id in runtime to avoid dead lock between blocks
		// because of data dependency
		bid_s = atomicAdd(blockCounter, 1);
	}
	__syncthreads();
	
	int bid = bid_s;
	// use shared memory
	__shared__ float shm[SECTION_SIZE];

	// fist stage: Thread block i first performs a scan on its scan block
	int idx = bid * blockDim.x + threadIdx.x;
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

	__syncthreads();

	// second stage: It then waits for its left neighbor block i-1 to pass the sum value.
	// Once it receives the sum from block i-1, it adds the value to its local sum 
	// and passes the cumulative sum value to its right neighbor block, i + 1.
	__shared__ float previous_block_scan_s;
	if (threadIdx.x == 0)
	{
		while(bid >= 1 && atomicAdd(flags + bid, 0) == 0) {} // wait data
		// read previous block scan sum
		previous_block_scan_s = blocks_scan[bid];
		// propagate current block scan sum that accumulate all previous blocks
		blocks_scan[bid + 1] = shm[blockDim.x - 1] + previous_block_scan_s;
		// memory fence
		__threadfence();
		// update current block flag to trigger right neighbor block to read
		atomicAdd(flags + bid + 1, 1);
	}
	__syncthreads();

	// third stage: It then moves on to add the sum value received from block i-1 to 
	// all partial scan values to produce all the output values of the scan block
	if (idx < len)
	{
		out[idx] = shm[threadIdx.x] + previous_block_scan_s;
	}
}

void single_pass_segmented_scan(float* input_h, int len, float* output_h) {
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

	float* S, *flags;
	cudaMalloc((void**)&S, size_S);
	cudaMalloc((void**)&flags, size_S);
	cudaMemset(S, 0, size_S);
	cudaMemset(flags, 0, size_S);

	dim3 block(SECTION_SIZE);
	dim3 grid(ceil(float(size_input) / SECTION_SIZE));
	printf("Launch segmented_kogge_stone_scan_kernel with grid: %d, block: %d\n", grid.x, block.x);
	segmented_kogge_stone_scan_kernel << <gird, block, 0, stream >> > (input_d, len, output_d, flags, S);

	cudaMemcpyAsync(output_h, output_d, size_output, cudaMemcpyDeviceToHost, stream);

	cudaStreamSynchronize(stream);

	cudaFree(input_d);
	cudaFree(output_d);
	cudaFree(S);

	cudaStreamDestroy(stream);
}