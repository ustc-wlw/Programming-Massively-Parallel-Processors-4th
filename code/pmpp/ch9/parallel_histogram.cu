#include <stdio.h>
#include <cuda_runtime.h>

#include "myutil.h"

static constexpr int CHAR_INTERVAL = 4; // a~d, e~h, i~l, m~p, q~t, u~x, y~z
static constexpr int NUM_BINS = ceil(26 / float(CHAR_INTERVAL));

static constexpr int CFACTOR = 4; // thread coarsening factor

void histogram_cpu(char* data, unsigned int len, unsigned int* histo) {
    for (size_t i = 0; i < len; i++)
    {
        // only count lower letter which between a~z
        if (data[i] >= 'a' && data[i] < 'z')
        {
            int diff = data[i] - 'a';
            histo[diff / CHAR_INTERVAL]++;
        }
    }
}


__global__
void histogram_atomic_kernel(char* data, unsigned int len, unsigned int* histo) {
    int idx = blockIdx.x + blockDim.x + threadIdx.x;
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
    int idx = blockIdx.x + blockDim.x + threadIdx.x;
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
    int idx = blockIdx.x + blockDim.x + threadIdx.x;

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


    int idx = blockIdx.x + blockDim.x + threadIdx.x;
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


    int idx = blockIdx.x + blockDim.x + threadIdx.x;
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
void histogram_thread_coarsening_interleaved_shm_privatization_atomic_kernel(char* data, unsigned int len, unsigned int* histo) {

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

    int idx = blockIdx.x + blockDim.x + threadIdx.x;
    // each thread caculate multiple interleaved input elements
    // the element interval is gridDim.x * blockDim.x = total threads number
    for (size_t i = idx; i < len; i += gridDim.x * blockDim.x)
    {
        int alphabet_position = data[i] - 'a';
        if (alphabet_position >= 0 && alphabet_position < 26)
        {
            // aggregation
            if (alphabet_position /  CHAR_INTERVAL == accumulator)
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

