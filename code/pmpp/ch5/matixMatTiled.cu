#include <stdio.h>
#include <cuda_runtime.h>

#include "myutil.h"

#define TILE_WIDTH 16

namespace ch5 {
enum matrix_shape_type {
        Base,                   // matrixMulKernelBase
        Multiple_Block_Square,  // matrixMulKernelFixSquare
        Generic_Square,         // matrixMulKernelGenericSquare
        Generic,                // matrixMulKernelGeneric
        Generic_Dynamic_Shm,    // matrixMulKernelGenericDynamicShm
};
}

using namespace ch5;

void matrixMulCPU(float* M, float* N, float* P, int M_height, int M_width, int N_width) {
    for (size_t row = 0; row < M_height; row++)
    {
        for (size_t col = 0; col < N_width; col++)
        {
            float pValue = 0;
            for (size_t k = 0; k < M_width; k++)
            {
                pValue += M[row * M_width + k] * N[k * N_width + col];
            }
            P[row * N_width + col] = pValue;
        }
    }
}

__device__ void printThread0(const char* info) {
    if(blockIdx.x == 0 && blockIdx.y == 0 
        && threadIdx.x ==0 && threadIdx.y == 0) {
        printf("%s\n", info);
    }
}

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
// block dim equals TILE_WIDTH
__global__
void matrixMulKernelFixSquare(float* M, float* N, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    float pValue = 0;
    for(int stage = 0; stage < (width / TILE_WIDTH); stage++) {
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
// block dim equals TILE_WIDTH
__global__
void matrixMulKernelGenericSquare(float* M, float* N, float* P, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    float pValue = 0;
    // caution: stage number change!!!
    for(int stage = 0; stage < ceil(width / float(TILE_WIDTH)); stage++) {
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

    // if(blockIdx.x == 0 && blockIdx.y == 0 
    //     && threadIdx.x ==0 && threadIdx.y == 0) {
    //     printf("M_height: %d, M_width: %d, N_width: %d, stage: %f\n", M_height, M_width, N_width,
    //             ceil(M_width / float(TILE_WIDTH)));
    // }

    // shared memory
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    float pValue = 0;
    for(int stage = 0; stage < ceil(M_width / float(TILE_WIDTH)); stage++) {
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
void matrixMulKernelGenericDynamicShm(float* M, float* N, float* P, int M_height,
        int M_width, int N_width, int M_shm_size, int N_shm_size) {

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // dynamic shared memory
    extern __shared__ char Mds_Nds[];
    float* Mds = (float*)(Mds_Nds);
    float* Nds = (float*)(Mds_Nds + M_shm_size);

    float pValue = 0;
    for(int stage = 0; stage < ceil(M_width / float(TILE_WIDTH)); stage++) {
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

void matMul2DTest(int blockSize, int M_height, int M_width, int N_width,
                matrix_shape_type shape_type) {

    unsigned int eleNum_M = M_height * M_width;
    unsigned int size_M = sizeof(float) * eleNum_M;
    unsigned int eleNum_N = M_width * N_width;
    unsigned int size_N = sizeof(float) * eleNum_N;
    unsigned int eleNum_P = M_height * N_width;
    unsigned int size_P = sizeof(float) * eleNum_P;

    float *h_M, *h_N, *h_P;
    // allocate pinned memory
    cudaMallocHost((void**)&h_M, size_M);
    cudaMallocHost((void**)&h_N, size_N);
    cudaMallocHost((void**)&h_P, size_P);

    float *hostRef = (float*) malloc(size_P);
    if(!hostRef) {
        printf("Can not allocat memory for hostRef!!!!\n");
    }
    memset(hostRef, 0, size_P);

    memset(h_P, 0, size_P);
    // initData(h_M, size_M);
    // initData(h_N, size_N);
    constantInit(h_M, size_M, 1);
    constantInit(h_N, size_N, 1);

    float *d_M, *d_N, *d_P;
    cudaMalloc((void**)&d_M, size_M);
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_P, size_P);

    cudaStream_t stream;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_M, h_M, size_M, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_N, h_N, size_N, cudaMemcpyHostToDevice, stream);

    dim3 block(blockSize, blockSize);
    dim3 grid((N_width + block.x - 1) / block.x, (M_height + block.y - 1) / block.y);
    printf("kernel launch setting: <<<grid(%d, %d), block(%d, %d)\n", grid.x, grid.y, block.x, block.y);

    cudaEventRecord(start, stream);
    int dynamic_shm_size = 0;
    int sub_shared_momery_size = 0;
    // launch kernel
    switch (shape_type)
    {
    case Base:
        matrixMulKernelBase<<<grid, block, 0, stream>>>(d_M, d_N, d_P, M_width);
        break;
    case Multiple_Block_Square:
        matrixMulKernelFixSquare<<<grid, block, 0, stream>>>(d_M, d_N, d_P, M_width);
        break;
    case Generic_Square:
        matrixMulKernelGenericSquare<<<grid, block, 0, stream>>>(d_M, d_N, d_P, M_width);
        break;
    case Generic:
        matrixMulKernelGeneric<<<grid, block, 0, stream>>>(d_M, d_N, d_P, M_height, M_width, N_width);
        break;
    case Generic_Dynamic_Shm:
        dynamic_shm_size = 2 * TILE_WIDTH * TILE_WIDTH * sizeof(float);
        sub_shared_momery_size = dynamic_shm_size / 2;
        printf("Test matrixMulKernelGenericDynamicShm with dynamic shared memory size: %d bytes!\n", dynamic_shm_size);
        matrixMulKernelGenericDynamicShm<<<grid, block, dynamic_shm_size, stream>>>(d_M, d_N, d_P, M_height, M_width,
                                N_width, sub_shared_momery_size, sub_shared_momery_size);
        break;
    default:
        printf("Invalid shape type, not support now!!!\n");
        exit(-1);
        break;
    }

    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    double flopsPerMatrixMul =
        2.0 * static_cast<double>(M_width) * static_cast<double>(M_height) * static_cast<double>(N_width);
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecTotal / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
           " WorkgroupSize= %u threads/block\n",
           gigaFlops, msecTotal, flopsPerMatrixMul, block.x * block.y);

    cudaMemcpyAsync(h_P, d_P, size_P, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    matrixMulCPU(h_M, h_N, hostRef, M_height, M_width, N_width);
    checkResult(hostRef, h_P, eleNum_P);

    // clear up memory
    cudaFreeHost(h_M);
    cudaFreeHost(h_N);
    cudaFreeHost(h_P);
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    free(hostRef);
}

int main() {

    int blockSize = TILE_WIDTH;

    // initial matrix dimension setting
    int M_width = 4 * blockSize;
    int M_height = M_width;
    int N_width = M_width;

    matrix_shape_type shape_type = Generic_Dynamic_Shm;
    switch (shape_type)
    {
        case Generic_Square:
            M_width += 3;
            M_height = M_width;
            N_width = M_width;
            break;
        case Generic:
        case Generic_Dynamic_Shm:
            M_width += 2;
            M_height += 3;
            N_width += 5;
            break;
        default:
            break;
    }

    printf("input Matrix M(%d,%d), Matrix N(%d,%d)\n", M_height, M_width, M_width, N_width);
    matMul2DTest(blockSize, M_height, M_width, N_width, shape_type);
}