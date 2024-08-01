#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>

#include "myutil.h"

static constexpr int THREAD_COARSENING_FACTOR = 2;
static constexpr int BLOCKDIM = 32;

enum class KERNEL_TYPE {
    SINGLE_BLOCK,
    REARANGE_SINGLE_BLOCK,
    SHM_REARANGE_SINGLE_BLOCK,
    MULTIBLOCK_SEGMENTED,
    THREAD_COARSEN_MULTIBLOCK_SEGMENTED,
};

float reduce_cpu(const float *data, unsigned int len) {
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++)
    {
        sum += data[i];
    }
    return sum;
}

// launch sole block, so maximun input data elements is 2048
// maximum thread block size is 1024
// assert input data length = blockDim.x * 2
// have two serious drawbacks: 
// 1、control divergence
// 2、memory divergence
__global__
void reduce_single_block_kernel(float *data, float *sumResult) {
    // int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx = threadIdx.x;
    int dataIndex = 2 * idx;

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (idx % stride == 0)
        {
            data[dataIndex] += data[dataIndex + stride];
        }
        __syncthreads();
    }

    if (idx == 0)
    {
        *sumResult = data[0];
    }
}

// thread index assignment to reduce thread block control divergence
// base on reduce_single_block_kernel
// assert input data length = blockDim.x * 2
// fix control divergence and control divergence probelm
__global__
void reduce_rearange_single_block_kernel(float *data, float *sumResult) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    for (int stride = blockDim.x; stride >= 1 && idx < stride; stride /= 2)
    {
        data[idx] += data[idx + stride];
        __syncthreads();
    }

    if (idx == 0)
    {
        *sumResult = data[0];
    }
}


// minimal global memory access by using shared memory
// besides, without change original input data
// assert input data length == blockDim.x * 2
__global__
void reduce_shm_rearange_single_block_kernel(const float *data, float *sumResult) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float input_s[BLOCKDIM];
    input_s[idx] = data[idx] + data[idx + blockDim.x];
    
    for (int stride = blockDim.x / 2; stride >= 1 && idx < stride; stride /= 2)
    {
        __syncthreads();
        input_s[idx] += input_s[idx + stride];
    }

    if (idx == 0)
    {
        *sumResult = input_s[0];
    }
}


// launch this kernel with multiple thread blocks, each block reduce 2 * blockDim.x elements
// atomicAdd each block partial reduce result to global result
__global__
void reduce_multiblock_segemented_kernel(const float *data, unsigned int len, float *sumResult) {
    // each block reduce 2 * blockDim.x elements
    // current thread data offset
    int idx = 2 * blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ float input_s[BLOCKDIM];
    if (idx < len && (idx + blockDim.x) < len)
    {
        input_s[tx] = data[idx] + data[idx + blockDim.x];
    }
    
    for (int stride = blockDim.x / 2; stride >= 1 && tx < stride; stride /= 2)
    {
        __syncthreads();
        input_s[tx] += input_s[tx + stride];
    }

    if (tx == 0)
    {
        // add each block partial reduce result to global result
        atomicAdd(sumResult, input_s[0]);
    }
}


// thread coarsening based on reduce_segemented_kernel to reduce thread blocks number
__global__
void reduce_thread_coarsening_multiblock_segemented_kernel(const float *data, unsigned int len, float *sumResult) {
    // each block reduce THREAD_COARSENING_FACTOR * 2 * blockDim.x elements
    // current thread data offset
    int idx = THREAD_COARSENING_FACTOR * 2 * blockDim.x * blockIdx.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ float input_s[BLOCKDIM];

    if (idx < len)
    {
        float sum = data[idx];
        // thread coarsening loop
        for (size_t i = 1; i < 2 * THREAD_COARSENING_FACTOR; i++)
        {
            if ((idx + i * blockDim.x) < len)
            {
                sum += data[idx + i * blockDim.x];
            }
        }
        input_s[tx] = sum;
        
        for (int stride = blockDim.x / 2; stride >= 1 && tx < stride; stride /= 2)
        {
            __syncthreads();
            input_s[tx] += input_s[tx + stride];
        }

        if (tx == 0)
        {
            // add each block partial reduce result to global result
            atomicAdd(sumResult, input_s[0]);
        }
    }
}

void parallel_reduce_kernel_test(float *data, unsigned int len, KERNEL_TYPE htype) {
    printf("test parallel_reduce kernel begain.....\n");
    int size_input = len * sizeof(float);
    int size_out = sizeof(float);
    printf("input string length is %d, total size is %d bytes\n", len, size_input);

    float reduceSum_h = 0;
    
    // for single block test, set block dim = input data length
    dim3 block(len);
    if (htype == KERNEL_TYPE::MULTIBLOCK_SEGMENTED ||
        htype == KERNEL_TYPE::THREAD_COARSEN_MULTIBLOCK_SEGMENTED)
    {
        block.x = BLOCKDIM;
    }
    dim3 grid((len + block.x - 1) / block.x);
    printf("Lauch kernel with grid(%d), block(%d) \n", grid.x, block.x);

    float *data_d;
    float *reduceSum_d;
    cudaMalloc((void**)&data_d, size_input);
    cudaMalloc((void**)&reduceSum_d, size_out);
    cudaMemcpy(data_d, data, size_input, cudaMemcpyHostToDevice);
    cudaMemset(reduceSum_d, 0, size_out);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    switch (htype)
    {
    case KERNEL_TYPE::SINGLE_BLOCK:
        reduce_single_block_kernel<<<grid, block>>>(data_d, reduceSum_d);
        break;
    case KERNEL_TYPE::REARANGE_SINGLE_BLOCK:
        reduce_rearange_single_block_kernel<<<grid, block>>>(data_d, reduceSum_d);
        break;
    case KERNEL_TYPE::SHM_REARANGE_SINGLE_BLOCK:
        reduce_shm_rearange_single_block_kernel<<<grid, block>>>(data_d, reduceSum_d);
        break;
    case KERNEL_TYPE::MULTIBLOCK_SEGMENTED:
        reduce_multiblock_segemented_kernel<<<grid, block>>>(data_d, len, reduceSum_d);
        break;
    case KERNEL_TYPE::THREAD_COARSEN_MULTIBLOCK_SEGMENTED:
        reduce_thread_coarsening_multiblock_segemented_kernel<<<grid, block>>>(data_d, len, reduceSum_d);
        break;
    default:
        printf("invalid kernel type!!!!\n");
        break;
    }

    cudaEventRecord(stop);
    cudaDeviceSynchronize();

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Time= %.3f msec, Size= %d chars, WorkgroupSize= %u threads/block\n",
            msecTotal, len, block.x * block.y);

    // D2H
    cudaMemcpy(&reduceSum_h, reduceSum_d, size_out, cudaMemcpyDeviceToHost);
    printf("**** the GPU result is: %f\n", reduceSum_h);

    float hostRef = reduce_cpu(data, len);
    printf("**** the CPU result is: %f\n", hostRef);
    checkResult(&hostRef, &reduceSum_h, 1);

    cudaFree(data_d);
    cudaFree(reduceSum_d);
}

int main(int argc, char** argv) {
    // set initial data
    std::vector<float> data(BLOCKDIM, 1);

    int kernel_type = 0;
    KERNEL_TYPE htype = KERNEL_TYPE::SINGLE_BLOCK;
    if (argc > 1)
    {
        kernel_type = int(*argv[1] - '0');
        printf("input kernel type is %d\n", kernel_type);
    }
    if (kernel_type >= 3)
    {
        // if test mutilblock kernel, increase input data size to more than 2*BLOCKDIM
        data.resize(4*BLOCKDIM + 22);
        std::fill(data.begin(), data.end(), 1);
        printf("input data size is %ld\n", data.size());
    }
    unsigned int len = data.size();

    switch (kernel_type)
    {
    case 0:
        htype = KERNEL_TYPE::SINGLE_BLOCK;
        break;
    case 1:
        htype = KERNEL_TYPE::REARANGE_SINGLE_BLOCK;
        break;
    case 2:
        printf("#### test reduce_shm_rearange_single_block_kernel ####\n");
        htype = KERNEL_TYPE::SHM_REARANGE_SINGLE_BLOCK;
        break;
    case 3:
        printf("#### test reduce_multiblock_segemented_kernel ####\n");
        htype = KERNEL_TYPE::MULTIBLOCK_SEGMENTED;
        break;
    case 4:
        printf("#### test reduce_thread_coarsening_multiblock_segemented_kernel ####\n");
        htype = KERNEL_TYPE::THREAD_COARSEN_MULTIBLOCK_SEGMENTED;
        break;
    default:
        printf("invalid kernel type!!!! use default kernel type 0\n");
        break;
    }

    parallel_reduce_kernel_test(data.data(), len, htype);

}