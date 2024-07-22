#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_WIDTH 16

__global__
void matrixMulKernelBase(float* M, float* N, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(col < width && row < width) {
        float dotSum = 0;
        for (size_t k = 0; k < width; k++)
        {
            dotSum += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = dotSum;
    }
}


// input matrix width is multiple of Block size and squre matrix
__global__
void matrixMulKernelFixSquare(float* M, float* N, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int stageNum = width / TILE_WIDTH;
    float pValue = 0;
    for(int stage = 0; stage < stageNum; stage++) {
        // load global data to shared memory by Tile (TILE_WIDTH * TILE_WIDTH per block)
        // each thread load a float
        Mds[ty][tx] = M[row * width + stage * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(stage * TILE_WIDTH + ty) * width + col];

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            pValue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    P[row * width + col] = pValue;
}


// input matrix width is arbitrary size and squre matrix
__global__
void matrixMulKernelGenericSquare(float* M, float* N, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int stageNum = width / TILE_WIDTH;
    float pValue = 0;
    for(int stage = 0; stage < stageNum; stage++) {
        // load global data to shared memory by Tile (TILE_WIDTH * TILE_WIDTH per block)
        // each thread load a float
        // add boundary check
        if((stage * TILE_WIDTH + tx) < width && row < width) {
            Mds[ty][tx] = M[row * width + stage * TILE_WIDTH + tx];
        }else{
            Mds[ty][tx] = 0;
        }
        if((stage * TILE_WIDTH + ty) < width && col < width) {
            Nds[ty][tx] = N[(stage * TILE_WIDTH + ty) * width + col];
        }else{
            Nds[ty][tx] = 0;
        }

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            pValue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    if(col < width && row < width) {
        P[row * width + col] = pValue;
    }
}


// input matrix width is arbitrary size
__global__
void matrixMulKernelGeneric(float* M, float* N, float* P, int M_height, int M_width, int N_width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int stageNum = M_width / TILE_WIDTH;
    float pValue = 0;
    for(int stage = 0; stage < stageNum; stage++) {
        // load global data to shared memory by Tile (TILE_WIDTH * TILE_WIDTH per block)
        // each thread load a float
        // add boundary check
        if((stage * TILE_WIDTH + tx) < M_width && row < M_height) {
            Mds[ty][tx] = M[row * M_width + stage * TILE_WIDTH + tx];
        }else{
            Mds[ty][tx] = 0;
        }
        // N_height = M_width
        if((stage * TILE_WIDTH + ty) < M_width && col < N_width) {
            Nds[ty][tx] = N[(stage * TILE_WIDTH + ty) * N_width + col];
        }else{
            Nds[ty][tx] = 0;
        }

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            pValue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    // output matrix shape: M_height * N_width
    if(col < N_width && row < M_height) {
        P[row * N_width + col] = pValue;
    }
}


// input matrix width is arbitrary size
// dynamic shared memory size
__global__
void matrixMulKernelGeneric(float* M, float* N, float* P, int M_height, int M_width, int N_width,
                            int M_shm_size, int N_shm_size) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // dynamic shared memory
    extern __shared__ char Mds_Nds[];
    float* Mds = (float*)(Mds_Nds);
    float* Nds = (float*)(Mds_Nds + M_shm_size);

    int stageNum = M_width / TILE_WIDTH;
    float pValue = 0;
    for(int stage = 0; stage < stageNum; stage++) {
        // load global data to shared memory by Tile (TILE_WIDTH * TILE_WIDTH per block)
        // each thread load a float
        // add boundary check
        if((stage * TILE_WIDTH + tx) < M_width && row < M_height) {
            Mds[ty * TILE_WIDTH + tx] = M[row * M_width + stage * TILE_WIDTH + tx];
        }else{
            Mds[ty * TILE_WIDTH + tx] = 0;
        }
        // N_height = M_width
        if((stage * TILE_WIDTH + ty) < M_width && col < N_width) {
            Nds[ty * TILE_WIDTH + tx] = N[(stage * TILE_WIDTH + ty) * N_width + col];
        }else{
            Nds[ty * TILE_WIDTH + tx] = 0;
        }

        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++) {
            pValue += Mds[ty * TILE_WIDTH + k] * Nds[k * TILE_WIDTH + tx];
        }

        __syncthreads();
    }

    // output matrix shape: M_height * N_width
    if(col < N_width && row < M_height) {
        P[row * N_width + col] = pValue;
    }
}