#include <stdio.h>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 3D grid and 3D seven-point stencil
// each output grid value is a weight sum of current grid and
// ORDER number values in each dim of x, y, z, total 3*(2 * ORDER) + 1
static constexpr int ORDER = 1;
static constexpr int neighbours = 6 * ORDER + 1;
static constexpr int IN_TILE_DIM = 32; // used for shared memory
static constexpr int OUT_TILE_DIM = IN_TILE_DIM - 2 * ORDER;

__constant__ float weights[neighbours];

/***************** all codes below based on ORDER is 1 ***************************/

//  The boundary cells contain boundary conditions that will
// not be updated from one iteration to the next.Thus only the inner output grid points need
// to be calculated during each stencil sweep
__global__
void stencil_sweep_basic_kernel(float* in, float* out, unsigned int N) {
	// input and output shape is same
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >=1 && x < N-1 && y >= 1 && y < N-1 && z >= 1 && z < N-1)
	{
		// ORDER = 1
		out[z * N * N + y * N + x] = weights[0] * in[z * N * N + y * N + x] +
			weights[1] * in[z * N * N + y * N + x - 1] + weights[2] * in[z * N * N + y * N + x + 1] + // x dim
			weights[3] * in[z * N * N + (y - 1) * N + x] + weights[4] * in[z * N * N + (y + 1) * N + x] + // y dim
			weights[5] * in[(z-1) * N * N + y * N + x] + weights[6] * in[(z+1) * N * N + y * N + x]; // z dim
	}
	else {
		// boundary grid value not change
		out[z * N * N + y * N + x] = in[z * N * N + y * N + x];
	}
}


// shared memory tile
// input tile is bigger than output tile
// block size is same as input tile
__global__
void stencil_sweep_3D_shared_memory_kernel(float* in, float* out, unsigned int N) {
	// input and output shape is same
	unsigned int x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - ORDER;
	unsigned int y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - ORDER;
	unsigned int z = blockIdx.z * OUT_TILE_DIM + threadIdx.z - ORDER;

	// 3D shared memory
	__shared__ float Ns[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];

	// load valid N data element to shared memory
	if (x >= 0 && x < N && y >= 0 && y < N && z >= 0 && z < N)
	{
		Ns[threadIdx.z][threadIdx.y][threadIdx.x] = in[z * N * N + y * N + x];
	}
	else {
		// ghost cell
		Ns[threadIdx.z][threadIdx.y][threadIdx.x] = 0;
	}

	__syncthreads();

	// only update inner output grid points
	if (x >= 1 && x < N - 1 && y >= 1 && y < N - 1 && z >= 1 && z < N - 1)
	{
		// deactivate ORDER exterior layer threads in current block
		if (threadIdx.x >= ORDER && threadIdx.x < IN_TILE_DIM - ORDER &&
			threadIdx.y >= ORDER && threadIdx.y < IN_TILE_DIM - ORDER &&
			threadIdx.z >= ORDER && threadIdx.z < IN_TILE_DIM - ORDER)
		{
			for (size_t k = 1; k <= ORDER; k++)
			{
				out[z * N * N + y * N + x] = weights[0] * Ns[threadIdx.z][threadIdx.y][threadIdx.x] +
					weights[1] * Ns[threadIdx.z][threadIdx.y][threadIdx.x - k] + weights[2] * Ns[threadIdx.z][threadIdx.y][threadIdx.x + k] + // x dim
					weights[3] * Ns[threadIdx.z][threadIdx.y - k][threadIdx.x] + weights[4] * Ns[threadIdx.z][threadIdx.y + k][threadIdx.x] + // y dim
					weights[5] * Ns[threadIdx.z - k][threadIdx.y][threadIdx.x] + weights[6] * Ns[threadIdx.z + k][threadIdx.y][threadIdx.x]; // z dim
			}
		}
	}
	
}



// thread coarsening
// iterate on the Z dim each time
__global__
void stencil_sweep_thread_coarsening_2D_shared_memory_kernel(float* in, float* out, unsigned int N) {
	unsigned int x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - ORDER;
	unsigned int y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - ORDER;
	unsigned int z_start = blockIdx.z * OUT_TILE_DIM;

	// define three 2D shared memory which contains x_y plane for preceeding, current and next in Z dim
	__shared__ float inPrev_s[IN_TILE_DIM][IN_TILE_DIM];
	__shared__ float inCurrent_s[IN_TILE_DIM][IN_TILE_DIM];
	__shared__ float inNext_s[IN_TILE_DIM][IN_TILE_DIM];

	// load previous x_y plane before current z dim  
	if (z_start - 1 >= 0 && z_start - 1 < N && x >= 0 && x < N && y >= 0 && y < N)
	{
		inPrev_s[threadIdx.y][threadIdx.x] = in[(z_start - 1) * N * N + y * N + x];
	}
	// load current x_y plane which z dim is z_start
	if (z_start >= 0 && z_start < N && x >= 0 && x < N && y >= 0 && y < N)
	{
		inCurrent_s[threadIdx.y][threadIdx.x] = in[z_start * N * N + y * N + x];
	}

	// thread coarsen and iterate on z dim
	for (size_t z_index = z_start; z_index < z_start + OUT_TILE_DIM; z_index++)
	{
		// load next x_y plane which z dim is (z_index + 1)
		if (z_index + 1 >= 0 && z_index + 1 < N && x >= 0 && x < N && y >= 0 && y < N)
		{
			inNext_s[threadIdx.y][threadIdx.x] = in[(z_index + 1) * N * N + y * N + x];
		}

		__syncthreads();

		// only update inner output grid points
		if (x >= 1 && x < N - 1 && y >= 1 && y < N - 1 && z_index >= 1 && z_index < N - 1) {
			// caculate output grid cell
			// deactivate ORDER exterior layer threads in current block
			if (threadIdx.x >= ORDER && threadIdx.x < IN_TILE_DIM - ORDER &&
				threadIdx.y >= ORDER && threadIdx.y < IN_TILE_DIM - ORDER &&
				threadIdx.z >= ORDER && threadIdx.z < IN_TILE_DIM - ORDER)
			{
				
				out[z_index * N * N + y * N + x] = weights[0] * inCurrent_s[threadIdx.y][threadIdx.x] +
					weights[1] * inCurrent_s[threadIdx.y][threadIdx.x - 1] + weights[2] * inCurrent_s[threadIdx.y][threadIdx.x + 1] + // x dim
					weights[3] * inCurrent_s[threadIdx.y - 1][threadIdx.x] + weights[4] * inCurrent_s[threadIdx.y + 1][threadIdx.x] + // y dim
					weights[5] * inPrev_s[threadIdx.y][threadIdx.x] + weights[6] * inNext_s[threadIdx.y][threadIdx.x]; // z dim
			}
		}

		__syncthreads();

		// update inPrev_s and inCurrent_s to reuse the loaded input elements
		inPrev_s[threadIdx.y][threadIdx.x] = inCurrent_s[threadIdx.y][threadIdx.x];
		inCurrent_s[threadIdx.y][threadIdx.x] = inNext_s[threadIdx.y][threadIdx.x];
	}
}


// register tiling
// thread coarsening
// iterate on the Z dim each time
__global__
void stencil_sweep_thread_coarsening_2D_shared_memory_kernel(float* in, float* out, unsigned int N) {
	unsigned int x = blockIdx.x * OUT_TILE_DIM + threadIdx.x - ORDER;
	unsigned int y = blockIdx.y * OUT_TILE_DIM + threadIdx.y - ORDER;
	unsigned int z_start = blockIdx.z * OUT_TILE_DIM;

	// define 2D shared memory which contains x_y plane of current block in every z dim
	__shared__ float inCurrent_s[IN_TILE_DIM][IN_TILE_DIM];
	// previous and next grid cell on the dim z is situated on register of each thread
	float inPrev;
	float inNext;

	// load previous grid cell before current z dim, wich coordinate in input data is (x, y, z_start -1) 
	if (z_start - 1 >= 0 && z_start - 1 < N && x >= 0 && x < N && y >= 0 && y < N)
	{
		inPrev = in[(z_start - 1) * N * N + y * N + x];
	}
	// load current x_y plane which z dim is z_start
	// the input elment loaded by this thread is (x, y, z dim)
	if (z_start >= 0 && z_start < N && x >= 0 && x < N && y >= 0 && y < N)
	{
		inCurrent_s[threadIdx.y][threadIdx.x] = in[z_start * N * N + y * N + x];
	}

	// thread coarsen and iterate on z dim
	for (size_t z_index = z_start; z_index < z_start + OUT_TILE_DIM; z_index++)
	{
		// load next x_y plane which z dim is (z_index + 1)
		if (z_index + 1 >= 0 && z_index + 1 < N && x >= 0 && x < N && y >= 0 && y < N)
		{
			inNext = in[(z_index + 1) * N * N + y * N + x];
		}

		__syncthreads();

		// only update inner output grid points
		if (x >= 1 && x < N - 1 && y >= 1 && y < N - 1 && z_index >= 1 && z_index < N - 1) {
			// caculate output grid cell
			// deactivate ORDER exterior layer threads in current block
			if (threadIdx.x >= ORDER && threadIdx.x < IN_TILE_DIM - ORDER &&
				threadIdx.y >= ORDER && threadIdx.y < IN_TILE_DIM - ORDER &&
				threadIdx.z >= ORDER && threadIdx.z < IN_TILE_DIM - ORDER)
			{

				out[z_index * N * N + y * N + x] = weights[0] * inCurrent_s[threadIdx.y][threadIdx.x] +
					weights[1] * inCurrent_s[threadIdx.y][threadIdx.x - 1] + weights[2] * inCurrent_s[threadIdx.y][threadIdx.x + 1] + // x dim
					weights[3] * inCurrent_s[threadIdx.y - 1][threadIdx.x] + weights[4] * inCurrent_s[threadIdx.y + 1][threadIdx.x] + // y dim
					weights[5] * inPrev + weights[6] * inNext; // z dim
			}
		}

		__syncthreads();

		// update inPrev_s and inCurrent_s to reuse the loaded input elements
		inPrev = inCurrent_s[threadIdx.y][threadIdx.x];
		inCurrent_s[threadIdx.y][threadIdx.x] = inNext;
	}
}