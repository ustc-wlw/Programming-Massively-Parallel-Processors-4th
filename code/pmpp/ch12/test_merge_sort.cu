#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <algorithm>

#include "merge_sort_kernel.h"
#include "myutil.h"

static constexpr int TILE_SIZE=256;
static constexpr int BLOCK_SIZE = 128;
static constexpr int GRID_SIZE = 16;

enum class KERNEL_TYPE {
    MERGE_BASIC,
    MERGE_TILED,
    CIRCULAR_BUFFER,
};

void merge_sort_kernel_test(const int* arrayA_h, unsigned int lenA,
                            const int* arrayB_h, unsigned int lenB,
                            KERNEL_TYPE htype) {
    int size_A = lenA * sizeof(int);
    int size_B = lenB * sizeof(int);
    int len_out = lenA + lenB;
    int size_out = len_out * sizeof(int);
    printf("input array A len: %d, array B len: %d, out array len: %d\n", lenA, lenB, len_out);

    int *arrayC_h, *array_ref;
    arrayC_h = (int*) malloc (size_out);
    if (!arrayC_h)
    {
        printf("malloc %d bytes for arrayC_h failed!\n", size_out);
    }
    array_ref = (int*) malloc (size_out);
    if (!array_ref)
    {
        printf("malloc %d bytes for array_ref failed!\n", size_out);
    }
    memset(array_ref, 0, size_out);
    memset(arrayC_h, 0, size_out);
    
    cudaEvent_t start, stop;
    cudaStream_t stream;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaStreamCreate(&stream));

    int *arrayA_d, *arrayB_d, *arrayC_d;
    CUDA_CHECK(cudaMalloc((void**)&arrayA_d, size_A));
    CUDA_CHECK(cudaMalloc((void**)&arrayB_d, size_B));
    CUDA_CHECK(cudaMalloc((void**)&arrayC_d, size_out));
    CUDA_CHECK(cudaMemcpyAsync(arrayA_d, arrayA_h, size_A, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(arrayB_d, arrayB_h, size_B, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemset(arrayC_d, 0, size_out));

    dim3 block(BLOCK_SIZE);
    // dim3 grid((len_out + block.x - 1) / block.x);
    dim3 grid(GRID_SIZE);
    printf("Lauch kernel with grid(%d), block(%d) \n", grid.x, block.x);


    CUDA_CHECK(cudaEventRecord(start, stream));
    switch (htype)
    {
    case KERNEL_TYPE::MERGE_BASIC:
        meger_basic_kernel<<<grid, block, 0, stream>>>(arrayA_d, lenA, arrayB_d, lenB, arrayC_d);
        break;
    case KERNEL_TYPE::MERGE_TILED:
        // merge_tiled_kernel<<<grid, block, 2 * TILE_SIZE, stream>>>(arrayA_d, lenA, arrayB_d, lenB, arrayC_d, TILE_SIZE);
        merge_tiled_kernel<<<grid, block, 0, stream>>>(arrayA_d, lenA, arrayB_d, lenB, arrayC_d, TILE_SIZE);
        break;
    case KERNEL_TYPE::CIRCULAR_BUFFER:
        merge_circular_buffer_kernel<<<grid, block, 0, stream>>>(arrayA_d, lenA, arrayB_d, lenB, arrayC_d, TILE_SIZE);
        break;
    default:
        printf("invalid kernel type!!!!\n");
        break;
    }

    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    

    float msecTotal = 0.0f;
    cudaEventElapsedTime(&msecTotal, start, stop);
    printf("Time= %.3f msec, total element Size= %d, WorkgroupSize= %u threads/block\n",
            msecTotal, len_out, block.x * block.y);

    // D2H
    CUDA_CHECK(cudaMemcpyAsync(arrayC_h, arrayC_d, size_out, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    printf("----------- cpu function start ------------\n");
    merge_sort_sequencial(arrayA_h, lenA, arrayB_h, lenB, array_ref);
    checkIntResult(array_ref, arrayC_h, len_out);

    CUDA_CHECK(cudaFree(arrayA_d));
    CUDA_CHECK(cudaFree(arrayB_d));
    CUDA_CHECK(cudaFree(arrayC_d));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaStreamDestroy(stream));

    free(arrayC_h);
    free(array_ref);
}

int main(int argc, char** argv) {
    int kernel_type = 0;
    KERNEL_TYPE htype = KERNEL_TYPE::MERGE_BASIC;

    unsigned int lenA = 100;
    unsigned int lenB = 200;
    if (argc > 2)
    {
        lenA = atoi(argv[1]);
        printf("input array A len is %d\n", lenA);
        lenB = atoi(argv[2]);
        printf("input array B len is %d\n", lenB);

        if (argc == 4)
        {
            kernel_type = atoi(argv[3]);
            printf("input kernel type is %d\n", kernel_type);
        }
    }

    switch (kernel_type)
    {
    case 0:
        htype = KERNEL_TYPE::MERGE_BASIC;
        printf("#### test meger_basic_kernel ####\n");
        break;
    case 1:
        printf("#### test merge_tiled_kernel ####\n");
        htype = KERNEL_TYPE::MERGE_TILED;
        break;
    case 2:
        printf("#### test merge_circular_buffer_kernel ####\n");
        htype = KERNEL_TYPE::CIRCULAR_BUFFER;
        break;
    default:
        printf("invalid kernel type!!!! use default kernel type 0\n");
        break;
    }

    std::vector<int> vA(lenA);
    std::vector<int> vB(lenB);

    initData_int(vA.data(), lenA, 0, 100);
    initData_int(vB.data(), lenB, 200, 700);

    std::sort(vA.begin(), vA.end());
    std::sort(vB.begin(), vB.end());

    printf("----------- array A ----------\n");
    for (size_t i = 0; i < 10; i++)
    {
        printf("%d, ", vA[i]);
    }
    printf("\n----------- array B ----------\n");
    for (size_t i = 0; i < 10; i++)
    {
        printf("%d, ", vB[i]);
    }
    printf("\n");
    

    merge_sort_kernel_test(vA.data(), lenA, vB.data(), lenB, htype);
}

