#pragma once

#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#define CUDA_CHECK(call)        \
{                               \
    const cudaError_t err=call; \
    if(err != cudaSuccess) {    \
        printf("%s in %s at line %d\n", cudaGetErrorString(err), \
        __FILE__, __LINE__);    \
        exit(EXIT_FAILURE);     \
    }                           \
}

double cpuSecond()
{
  struct timeval tp;
  gettimeofday(&tp,NULL);
  return((double)tp.tv_sec+(double)tp.tv_usec*1e-6);

}

void initData(float* data, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (size_t i = 0; i < size; i++)
    {
        data[i] = (float)(rand() & 0xffff) / 1000.0f;
        // data[i] = 1;
    }
}

void initData_int(float* data, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (size_t i = 0; i < size; i++)
    {
        data[i] = (rand() & 0xff) % 100;
    }
}

void checkResult(float* hostRef, float* gpuRef, const int N) {
    double epsilon=1.0E-8;
    for(int i = 0; i < N; i++)
    {
        if(abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            printf("Results don\'t match!\n");
            printf("%f(hostRef[%d]) != %f(gpuRef[%d])\n", hostRef[i], i, gpuRef[i], i);
            return;
        }
    }
    printf("Check result success!\n");
}