#include <cuda_runtime.h>
#include <stdio.h>

#include "myutil.h"

static constexpr int FILTER_RADIUS = 2;
static constexpr int IN_TILE_DIM = 32; // Input Tile dim
static constexpr int OUT_TILE_DIM = IN_TILE_DIM - 2 * FILTER_RADIUS; // Output Tile dim
static constexpr int TILE_DIM = 32;

enum convolution_type {
    BASIC,
    CONST_MEMORY,
    TILED_CONST_MEMORY_1,
    TILED_CONST_MEMORY_2,
    CACHED_TILED_CONST_MEMORY,
};

__constant__ float filter[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];


void convolution_2D_cpu(float *N, float *filter, float *P,
                                int radius, int width, int height) {
    // input and output matrix shape: height * width
    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < width; col++)
        {
            float pValue = 0;
            // iterate kernel and according input matrix path
            for (size_t fRow = 0; fRow < 2 * radius + 1; fRow++)
            {
                for (size_t fCol = 0; fCol < 2 * radius + 1; fCol++)
                {
                    int inRow = row - radius + fRow;
                    int inCol = col - radius + fCol;
                    if(inRow >= 0 && inRow < height && inCol < width && inCol >= 0) {
                        pValue += filter[fRow * radius + fCol] * N[inRow * width + inCol];
                    }
                }
            }
            P[row * width + col] = pValue;
        }
    }
}


__global__
void convolution_2D_basic_kernel(float *N, float *filter, float *P,
                                int radius, int width, int height) {
    // input and output matrix shape: height * width
    // each thread is responsible for caculating a element
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float pValue = 0;
    // iterate kernel and according input matrix path
    for (size_t fRow = 0; fRow < 2 * radius + 1; fRow++)
    {
        for (size_t fCol = 0; fCol < 2 * radius + 1; fCol++)
        {
            int inRow = row - radius + fRow;
            int inCol = col - radius + fCol;
            if(inRow >= 0 && inRow < height && inCol < width && inCol >= 0) {
                pValue += filter[fRow * radius + fCol] * N[inRow * width + inCol];
            }
        }
        
    }
    if (row < height && col < width) {
        P[row * width + col] = pValue;
    }
}


// load filter weight from const memory
__global__
void convolution_2D_with_constMemory_kernel(float *N, float *P, int width, int height) {
    // input and output matrix shape: height * width
    // each thread is responsible for caculating a element
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // if (col ==0 && row ==0)
    // {
    //     for (size_t i = 0; i < 5; i++)
    //     {
    //         for (size_t j = 0; j < 5; j++)
    //         {
    //             printf("%f, ", filter[i][j]);
    //         }
    //     }
    //     printf("\n");
    // }

    float pValue = 0;
    // iterate kernel and according input matrix path
    for (size_t fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++)
    {
        for (size_t fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++)
        {
            int inRow = row - FILTER_RADIUS + fRow;
            int inCol = col - FILTER_RADIUS + fCol;
            if(inRow >= 0 && inRow < height && inCol < width && inCol >= 0) {
                // filter data situated in const memory
                pValue += filter[fRow][fCol] * N[inRow * width + inCol];
            }
        }
        
    }
    if (row < height && col < width) {
        P[row * width + col] = pValue;
    }
}

// load filter weight from const memory
// load matrix N from HBM to shared memory in Tile
// block dim is same as Input Tail(which is Ns in code below)
// Input Tail size = Output Tail size + 2 * FILTER_RADIUS
__global__
void convolution_2D_Tiled_constMemory_kernel1(float *N, float *P, int width, int height) {
    // col and row which is responsible for caculate output element of P
    // and load input element of N, N shape is same as P
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // shared memory to store input tile of matrix N
    __shared__ float Ns[IN_TILE_DIM][IN_TILE_DIM];
    if(row >= 0 and row < height && col >= 0 and col < width) {
        Ns[ty][tx] = N[row * width + col];
    }else {
        Ns[ty][tx] = 0;
    }

    __syncthreads();

    // upper left corner coordinate which belongs to the patch that caculate with filter
    int tileCol = tx - FILTER_RADIUS;
    int tileRow = ty - FILTER_RADIUS;

    // deactivate FILTER_RADIUS exterior layers of threads
    if (col >= 0 && col < width && row >= 0 && row < height)
    {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM &&
            tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float pValue = 0;
            for (size_t fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++)
            {
                for (size_t fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    pValue += Ns[tileRow + fRow][tileCol + fCol] * filter[fRow][fCol];
                }
            }
            P[row * width + height] = pValue;
        }
    }
}


// load filter weight from const memory
// load matrix N from HBM to shared memory in Tile
// block dim is same as Output Tail
// Input Tail size = Output Tail size + 2 * FILTER_RADIUS
__global__
void convolution_2D_Tiled_constMemory_kernel2(float *N, float *P, int width, int height) {
    // col and row which is responsible for caculating output element of P
    // and load element of input matrix N, N shape is same as output matrix P
    int col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;
    int row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // shared memory to store input tile of matrix N
    __shared__ float Ns[IN_TILE_DIM][IN_TILE_DIM];
    if(row >= 0 and row < height && col >= 0 and col < width) {
        Ns[ty][tx] = N[row * width + col];
    }else {
        Ns[ty][tx] = 0;
    }

    __syncthreads();

    // upper left corner coordinate which belongs to the patch that caculate with filter
    int tileCol = tx;
    int tileRow = ty;

    // deactivate FILTER_RADIUS exterior layers of threads
    if (col >= 0 && col < width && row >= 0 && row < height)
    {
        float pValue = 0;
        for (size_t fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++)
        {
            for (size_t fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                pValue += Ns[tileRow + fRow][tileCol + fCol] * filter[fRow][fCol];
            }
        }
        P[row * width + height] = pValue;
    }
}


// load filter weight from const memory
// load matrix N in HBM to shared memory in Tile
// block dim is same as Input Tail(which is Ns in code below)
// Input Tail size = Output Tail size
// using L2 cache for halo cells (surrounding elements to caculate current ouput element of P)
__global__
void convolution_2D_cached_Tiled_constMemory_kernel(float *N, float *P, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // shared memory for Input Tile
    // only containing part internal elements in matrix N
    // some internal elemnts which are called halo cells are cached in L2
    __shared__ float Ns[TILE_DIM][TILE_DIM];
    if (col >= 0 && col < width && row >= 0 && row < height)
    {
        Ns[ty][tx] = N[row * width + col];
    } else {
        Ns[ty][tx] = 0;
    }
    
    __syncthreads();

    // caculate output element
    // deactivate the threads at the edges of the block
    if (col < width && row < height)
    {
        float pValue = 0;
        // caculate output element based on Input Tile, halo cells, and ghost cells
        for (size_t fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++)
        {
            for (size_t fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                // Input Tile elements
                if (tx - FILTER_RADIUS + fCol >= 0 && tx - FILTER_RADIUS + fCol < TILE_DIM &&
                    ty - FILTER_RADIUS + fRow >= 0 && ty - FILTER_RADIUS + fRow < TILE_DIM)
                {
                    pValue += filter[fRow][fCol] * Ns[ty - FILTER_RADIUS + fRow][tx - FILTER_RADIUS + fCol];
                }
                // halo cells
                else if (col - FILTER_RADIUS + fCol >= 0 && col - FILTER_RADIUS + fCol < width &&
                        row - FILTER_RADIUS + fRow >= 0 && row - FILTER_RADIUS + fRow < height)
                {
                    // load from L2 cache
                    pValue += filter[fRow][fCol] * N[width * (row - FILTER_RADIUS + fRow) + col - FILTER_RADIUS + fCol];
                }
                // ghost cell dont care
            }
        }

        P[row * width + col] = pValue;
    }
}


void matMul2DTest(int blockSize, int height, int width,
                convolution_type cov_type) {

    unsigned int eleNum_N = height * width;
    unsigned int size_N = sizeof(float) * eleNum_N;
    unsigned int filter_dim = 2 * FILTER_RADIUS + 1;
    unsigned int eleNum_filter =  pow(filter_dim, 2);
    unsigned int size_filter = sizeof(float) * eleNum_filter;
    unsigned int size_P = size_N;

    printf("input Matrix N(%d,%d), filter(%d,%d), output Matrix P(%d,%d)\n", 
            height, width, filter_dim, filter_dim, height, width);

    float *h_N, *h_P;
    // allocate pinned memory
    cudaMallocHost((void**)&h_N, size_N);
    cudaMallocHost((void**)&h_P, size_P);

    float *hostRef = (float*) malloc(size_P);
    if(!hostRef) {
        printf("Can not allocat memory for hostRef!!!!\n");
    }
    memset(hostRef, 0, size_P);

    float *h_filter = (float*) malloc(size_P);
    if(!h_filter) {
        printf("Can not allocat memory for h_filter!!!!\n");
    }
    constantInit(h_filter, eleNum_filter, 1.0f);

    memset(h_P, 0, size_P);
    // initData(h_M, size_M);
    // initData(h_N, size_N);
    constantInit(h_N, size_N, 1);

    float *d_N, *d_P;
    cudaMalloc((void**)&d_N, size_N);
    cudaMalloc((void**)&d_P, size_P);

    cudaStream_t stream;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaStreamCreate(&stream);

    // copy host data to const memory in HBM
    cudaMemcpyToSymbol(filter, h_filter, size_filter);
    // copy host data to HBM
    cudaMemcpyAsync(d_N, h_N, size_N, cudaMemcpyHostToDevice, stream);

    dim3 block(blockSize, blockSize);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    printf("kernel launch setting: <<<grid(%d, %d), block(%d, %d)\n", grid.x, grid.y, block.x, block.y);

    cudaEventRecord(start, stream);
    // launch kernel
    switch (cov_type)
    {
    case CONST_MEMORY:
        convolution_2D_with_constMemory_kernel<<<grid, block, 0, stream>>>(d_N, d_P, width, height);
        break;
    case TILED_CONST_MEMORY_1:
        convolution_2D_Tiled_constMemory_kernel1<<<grid, block, 0, stream>>>(d_N, d_P, width, height);
        break;
    case TILED_CONST_MEMORY_2:
        convolution_2D_Tiled_constMemory_kernel2<<<grid, block, 0, stream>>>(d_N, d_P, width, height);
        break;
    case CACHED_TILED_CONST_MEMORY:
        convolution_2D_cached_Tiled_constMemory_kernel<<<grid, block, 0, stream>>>(d_N, d_P, width, height);
    default:
        printf("Invalid shape type, not support now!!!\n");
        exit(-1);
        break;
    }

    cudaEventRecord(stop, stream);

    cudaEventSynchronize(stop);

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    double flops =
        2.0 * static_cast<double>(eleNum_filter) * static_cast<double>(height) * static_cast<double>(width);
    double gigaFlops = (flops * 1.0e-9f) / (msecTotal / 1000.0f);
    printf("Performance= %.2f GFlop/s, Time= %.3f msec, Size= %.0f Ops,"
           " WorkgroupSize= %u threads/block\n",
           gigaFlops, msecTotal, flops, block.x * block.y);

    cudaMemcpyAsync(h_P, d_P, size_P, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    convolution_2D_cpu(h_N, h_filter, hostRef, FILTER_RADIUS, width, height);
    checkResult(hostRef, h_P, eleNum_N);

    // clear up memory
    cudaFreeHost(h_N);
    cudaFreeHost(h_P);
    cudaFree(d_N);
    cudaFree(d_P);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    free(hostRef);
    free(h_filter);
}

int main() {

    int blockSize = 32;

    // initial matrix dimension setting
    int height = 4 * blockSize;
    int width = height;

    convolution_type ctype = TILED_CONST_MEMORY_1;

    
    matMul2DTest(blockSize, height, width, ctype);
}