#include <stdio.h>
#include <cuda_runtime.h>

#include "myutil.h"

static constexpr int CHAR_INTERVAL = 4; // a~d, e~h, i~l, m~p, q~t, u~x, y~z
static constexpr int NUM_BINS = ceil(26 / float(CHAR_INTERVAL));

static constexpr int CFACTOR = 4; // thread coarsening factor

enum class HISTO_TYPE {
    ATOMIC,
    GLOBAL_PRIVATIZATION_ATOMIC,
    SHARED_PRIVATIZATION_ATOMIC,
    THREAD_COARSENING_CONTIGUOUS,
    THREAD_COARSENING_INTERLEAVING,
    AGGREGATION,
};

void histogram_cpu(const char* data, unsigned int len, unsigned int* histo) {
    printf("input string len is %d, data address is %p\n", len, data);
    for (int i = 0; i < len; i++)
    {
        // only count lower letter which between a~z
        if (data[i] >= 'a' && data[i] <= 'z')
        {
            int diff = data[i] - 'a';
            histo[diff / CHAR_INTERVAL]++;
        }
    }
}


__global__
void histogram_atomic_kernel(char* data, unsigned int len, unsigned int* histo) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        int alphabet_position = data[idx] - 'a';
        // only count lower letter which between a~z
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            atomicAdd(histo + alphabet_position /  CHAR_INTERVAL, 1);
        }
    }

}


// a private copy of histo for each thread block
// input param histo is located in HBM, and size = gridDim.x * NUM_BINS * sizeof(int)
// also maybe in L2 cache
__global__
void histogram_global_privatization_atomic_kernel(char* data, unsigned int len, unsigned int* histo) {
    // each thread block atomic update according private histo
    // whose global start address is histo + blockIdx.x * NUM_BINS
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len)
    {
        int alphabet_position = data[idx] - 'a';
        // only count lower letter which between a~z
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            atomicAdd(histo + blockIdx.x * NUM_BINS + alphabet_position /  CHAR_INTERVAL, 1);
        }
    }

    // after this thread block finishing updating private histo
    // merge into block 0 private histo as the final result
    if (blockIdx.x > 0)
    {
        __syncthreads();
        // in case that NUM_BINS is larger than blockSize
        for (size_t binIndex = threadIdx.x; binIndex < NUM_BINS; binIndex += blockDim.x)
        {
            // read private histo value updated by this thread block
            int binValue = histo[blockIdx.x * NUM_BINS + binIndex];
            if (binValue > 0)
            {
                atomicAdd(histo + binIndex, binValue);
            }
        }
    }
}


// a private copy of histo for each thread block
// the private copy is situatied in shared memory
// input param histo is located in HBM, and size = NUM_BINS * sizeof(int)
// "shm" stands for "shared memory"
__global__
void histogram_shm_privatization_atomic_kernel(char* data, unsigned int len, unsigned int* histo) {
    // each thread block atomic update according private histo
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int histo_s[NUM_BINS];
    // initialize shared memory
    // in case that NUM_BINS is larger than blockSize
    for (size_t i = threadIdx.x; i < NUM_BINS; i+= blockDim.x)
    {
        histo_s[i] = 0;
    }

    __syncthreads();

    if (idx < len)
    {
        int alphabet_position = data[idx] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            // update private histo in shared memory
            atomicAdd(histo_s + alphabet_position /  CHAR_INTERVAL, 1);
        }
    }

    __syncthreads();

    // after this thread block finishing updating private histo
    // merge into public global histo which is the input param "histo"

    // in case that NUM_BINS is larger than blockSize
    for (size_t binIndex = threadIdx.x; binIndex < NUM_BINS; binIndex += blockDim.x)
    {
        // read private histo value in shared memory
        int binValue = histo_s[binIndex];
        if (binValue > 0)
        {
            atomicAdd(histo + binIndex, binValue);
        }
    }
}


// thread coarsening based on histogram_shm_privatization_atomic_kernel
// each thread caculate contiguous multiple characters to reduce thread block numbers
__global__
void histogram_thread_coarsening_contiguous_shm_privatization_atomic_kernel(char* data, unsigned int len, unsigned int* histo) {

    __shared__ int histo_s[NUM_BINS];
    // initialize shared memory
    // in case that NUM_BINS is larger than blockSize
    for (size_t i = threadIdx.x; i < NUM_BINS; i+= blockDim.x)
    {
        histo_s[i] = 0;
    }

    __syncthreads();


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread caculate contiguous input elements, whose start address is (data + idx * CFACTOR)
    // end address is data + min((idx+1) * CFACTOR, len)
    for (size_t i = idx * CFACTOR; i < min((idx+1) * CFACTOR, len); i++)
    {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            // update private histo in shared memory
            atomicAdd(histo_s + alphabet_position /  CHAR_INTERVAL, 1);
        }
    }
    
    __syncthreads();

    // after this thread block finishing updating private histo
    // merge into public global histo which is the input param "histo"

    // in case that NUM_BINS is larger than blockSize
    for (size_t binIndex = threadIdx.x; binIndex < NUM_BINS; binIndex += blockDim.x)
    {
        // read private histo value in shared memory
        int binValue = histo_s[binIndex];
        if (binValue > 0)
        {
            atomicAdd(histo + binIndex, binValue);
        }
    }
}


// thread coarsening based on histogram_shm_privatization_atomic_kernel
// each thread caculate interleaved multiple characters to reduce thread block numbers
// the element interval is gridDim.x * blockDim.x = total threads number
__global__
void histogram_thread_coarsening_interleaved_shm_privatization_atomic_kernel(char* data, unsigned int len, unsigned int* histo) {

    __shared__ int histo_s[NUM_BINS];
    // initialize shared memory
    // in case that NUM_BINS is larger than blockSize
    for (size_t i = threadIdx.x; i < NUM_BINS; i+= blockDim.x)
    {
        histo_s[i] = 0;
    }

    __syncthreads();


    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread caculate multiple interleaved input elements
    // the element interval is gridDim.x * blockDim.x = total threads number
    for (size_t i = idx; i < len; i += gridDim.x * blockDim.x)
    {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            // update private histo in shared memory
            atomicAdd(histo_s + alphabet_position /  CHAR_INTERVAL, 1);
        }
    }
    
    __syncthreads();

    for (size_t binIndex = threadIdx.x; binIndex < NUM_BINS; binIndex += blockDim.x)
    {
        int binValue = histo_s[binIndex];
        if (binValue > 0)
        {
            atomicAdd(histo + binIndex, binValue);
        }
    }
}


// atomic operator aggregation based on histogram_thread_coarsening_interleaved_shm_privatization_atomic_kernel
// if severl sequential atomic ops are updating the same address, it is prefer to accumulate the severl updates
// to a single atomic operation
__global__
void histogram_aggregation_thread_coarsening_interleaved_shm_privatization_atomic_kernel(char* data, unsigned int len, unsigned int* histo) {

    __shared__ int histo_s[NUM_BINS];
    // initialize shared memory
    // in case that NUM_BINS is larger than blockSize
    for (size_t i = threadIdx.x; i < NUM_BINS; i+= blockDim.x)
    {
        histo_s[i] = 0;
    }

    __syncthreads();

    unsigned int accumulator = 0;
    int preBinIdx = -1; 

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // each thread caculate multiple interleaved input elements
    // the element interval is gridDim.x * blockDim.x = total threads number
    for (size_t i = idx; i < len; i += gridDim.x * blockDim.x)
    {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            // aggregation
            if (alphabet_position /  CHAR_INTERVAL == preBinIdx)
            {
                accumulator++;
            } else {
                // current bin_index is diffirent from previous one
                // so atomic update previous bin and change current state
                if (accumulator > 0)
                {
                    atomicAdd(histo_s + preBinIdx, accumulator);
                }
                accumulator = 1;
                preBinIdx = alphabet_position /  CHAR_INTERVAL;
            }
        }
    }

    // make sure all update
    if (accumulator > 0)
    {
        atomicAdd(histo_s + preBinIdx, accumulator);
    }
    
    __syncthreads();

    for (size_t binIndex = threadIdx.x; binIndex < NUM_BINS; binIndex += blockDim.x)
    {
        int binValue = histo_s[binIndex];
        if (binValue > 0)
        {
            atomicAdd(histo + binIndex, binValue);
        }
    }
}

void histogram_kernel_test(const char *data, unsigned int len, HISTO_TYPE htype) {
    printf("test histogram kernel begain.....\n");
    int size_input = len * sizeof(char);
    int size_histo = NUM_BINS * sizeof(unsigned int);
    printf("input string length is %d, total size is %d bytes\n", len, size_input);
    printf("total histogram bins number:  %d\n", NUM_BINS);

    unsigned int *hostRef = (unsigned int*) malloc(size_histo);
    if (!hostRef)
    {
        printf("Alloc hostRef memory failed!!\n");
        exit(-1);
    }
    unsigned int *histo_h = (unsigned int*) malloc(size_histo);
    if (!histo_h)
    {
        printf("Alloc histo_h memory failed!!\n");
        exit(-1);
    }
    memset(hostRef, 0, size_histo);
    memset(histo_h, 0, size_histo);
    
    dim3 block(32);
    dim3 grid((len + block.x - 1) / block.x);
    printf("Lauch kernel with grid(%d), block(%d) \n", grid.x, block.x);

    char *data_d;
    unsigned int *histo_d;
    cudaMalloc((void**)&data_d, size_input);
    // H2D
    cudaMemcpy(data_d, data, size_input, cudaMemcpyHostToDevice);
    if (htype == HISTO_TYPE::GLOBAL_PRIVATIZATION_ATOMIC)
    {
        cudaMalloc((void**)&histo_d, size_histo * grid.x);
        cudaMemset(histo_d, 0, size_histo * grid.x);
    } else {
        cudaMalloc((void**)&histo_d, size_histo);
        cudaMemset(histo_d, 0, size_histo);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    switch (htype)
    {
    case HISTO_TYPE::ATOMIC:
        // printf("histogram_atomic_kernel run!!!\n");
        histogram_atomic_kernel<<<grid, block>>>(data_d, len, histo_d);
        break;
    case HISTO_TYPE::GLOBAL_PRIVATIZATION_ATOMIC:
        histogram_global_privatization_atomic_kernel<<<grid, block>>>(data_d, len, histo_d);
        break;
    case HISTO_TYPE::SHARED_PRIVATIZATION_ATOMIC:
        histogram_shm_privatization_atomic_kernel<<<grid, block>>>(data_d, len, histo_d);
        break;
    case HISTO_TYPE::THREAD_COARSENING_CONTIGUOUS:
        histogram_thread_coarsening_contiguous_shm_privatization_atomic_kernel<<<grid, block>>>(data_d, len, histo_d);
        break;
    case HISTO_TYPE::THREAD_COARSENING_INTERLEAVING:
        histogram_thread_coarsening_interleaved_shm_privatization_atomic_kernel<<<grid, block>>>(data_d, len, histo_d);
        break;
    case HISTO_TYPE::AGGREGATION:
        histogram_aggregation_thread_coarsening_interleaved_shm_privatization_atomic_kernel<<<grid, block>>>(data_d, len, histo_d);
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
    cudaMemcpy(histo_h, histo_d, size_histo, cudaMemcpyDeviceToHost);
    printf("after kernel finish, the kernel result is: \n");
    for (int i = 0; i < NUM_BINS; i++)
    {
        printf("Bin %d value: %d\n", i, histo_h[i]);
    }
    

    histogram_cpu(data, len, hostRef);
    printf("the CPU result is: \n");
    for (int i = 0; i < NUM_BINS; i++)
    {
        printf("Bin %d value: %d\n", i, hostRef[i]);
    }
    checkIntResult(hostRef, histo_h, NUM_BINS);

    cudaFree(data_d);
    cudaFree(histo_d);
    free(hostRef);
    free(histo_h);
}

int main(int argc, char** argv) {
    char input_text[] = "Gather the embeddings weight for initialization. DeepSpeed will automatically gather a module parameters during its constructor and for its forward and backward pass. However, additional accesses must coordinate with DeepSpeed to ensure that parameter data is gathered and subsequently partitioned. If the tensor is modified, the modifier_rank argument should also be used to ensure all ranks have a consistent view of the data. Please see the full GatheredParameters docs for more details.";
    unsigned int len = strlen(input_text) / sizeof(input_text[0]);

    int kernel_type = 0;
    HISTO_TYPE htype = HISTO_TYPE::ATOMIC;
    if (argc > 1)
    {
        kernel_type = int(*argv[1] - '0');
        printf("input kernel type is %d\n", kernel_type);
    }

    switch (kernel_type)
    {
    case 0:
        htype = HISTO_TYPE::ATOMIC;
        break;
    case 1:
        htype = HISTO_TYPE::GLOBAL_PRIVATIZATION_ATOMIC;
        break;
    case 2:
        htype = HISTO_TYPE::SHARED_PRIVATIZATION_ATOMIC;
        break;
    case 3:
        htype = HISTO_TYPE::THREAD_COARSENING_CONTIGUOUS;
        break;
    case 4:
        htype = HISTO_TYPE::THREAD_COARSENING_INTERLEAVING;
        break;
    case 5:
        htype = HISTO_TYPE::AGGREGATION;
        break;
    default:
        printf("invalid kernel type!!!! use default kernel type 0\n");
        break;
    }

    histogram_kernel_test(input_text, len, htype);

}