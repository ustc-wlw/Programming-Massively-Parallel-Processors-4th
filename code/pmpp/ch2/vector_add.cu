#include "cuda_runtime.h"
#include <stdio.h>

#include "myutil.h"

void vecAdd_cpu(float* A_h, float* B_h, float* C_h, int n) {
    for (size_t i = 0; i < n; i++)
    {
        C_h[i] = A_h[i] + B_h[i];
    }
}

__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < n) {
        C[tid] = A[tid] + B[tid];
    }
}

int main() {
    float *A_h, *B_h, *C_h;
    float *hostRef;
    int eleNum = 1 << 6;
    size_t totalSize = sizeof(float) * eleNum;
    printf("input data number: %d, total size: %ld\n", eleNum, totalSize);
    A_h = (float*) malloc(totalSize);
    B_h = (float*) malloc(totalSize);
    C_h = (float*) malloc(totalSize);
    hostRef = (float*) malloc(totalSize);

    // init input data
    memset(C_h, 0, totalSize);
    memset(hostRef, 0, totalSize);
    initData(A_h, eleNum);
    initData(B_h, eleNum);

    // Allocate device memory
    float *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, totalSize);
    cudaMalloc((void**)&B_d, totalSize);
    cudaMalloc((void**)&C_d, totalSize);

    // copy data from host to device
    cudaMemcpy(A_d, A_h, totalSize, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, totalSize, cudaMemcpyHostToDevice);

    // launch kernel
    dim3 blockDim(128); // 1D thread block
    dim3 gridDim(ceil(eleNum / float(blockDim.x))); // 1D grid
    printf("Kernel Execution configuration<<<%d,%d>>>\n", gridDim.x, blockDim.x);
    vecAddKernel<<<gridDim, blockDim>>>(A_d, B_d, C_d, eleNum);

    // copy result data from device to Host
    // synchronize to wait kernel finish
    cudaMemcpy((void**)C_h, (void**)(C_d), totalSize, cudaMemcpyDeviceToHost);

    // call cpu function
    vecAdd_cpu(A_h, B_h, hostRef, eleNum);
    checkResult(hostRef, C_h, eleNum);

    // free device memory
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);

    // free host memory
    free(A_h);
    free(B_h);
    free(C_h);
    free(hostRef);
}