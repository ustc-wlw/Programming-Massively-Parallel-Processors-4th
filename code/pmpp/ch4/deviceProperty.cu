#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int devCount;
    cudaGetDeviceCount(&devCount);
    printf("Current GPU device number: %d\n", devCount);

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, 0);
    printf("Current device name: %s\n", devProp.name);
    printf("Current device maxThreadsPerBlock: %d\n", devProp.maxThreadsPerBlock);
    printf("Current device SM number: %d\n", devProp.multiProcessorCount);
    printf("Current device clockRate: %d\n", devProp.clockRate);
    printf("Current device max Thread in Block for each dim: %d, %d, %d\n", 
            devProp.maxThreadsDim[0], devProp.maxThreadsDim[1], devProp.maxThreadsDim[2]);
    printf("Current device max blocks in Grid for each dim: %d, %d, %d\n", 
            devProp.maxGridSize[0], devProp.maxGridSize[1], devProp.maxGridSize[2]);
    printf("register available each SM: %d\n", devProp.regsPerBlock);
    printf("Current device warp size: %d\n", devProp.warpSize);

}