#include <cuda_runtime.h>

#define RGB_CHANNEL 3
#define BLUR_SIZE 2

__global__ 
void color2GrayConvertion(unsigned char* Pout,
            unsigned char* Pin, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < width && row < height) {
        // get 1D offset for the grayScale image
        int grayOffset = row * width + col;
        // get 1D offset for the rgb image
        int rgbOffset = grayOffset * RGB_CHANNEL;
        unsigned char r = Pin[rgbOffset];
        unsigned char g = Pin[rgbOffset + 1];
        unsigned char b = Pin[rgbOffset + 2];
        Pout[grayOffset] = 0.21f * r + 0.71f * g + 0.07f * b;
    }
}

__global__
void blurKernel(unsigned char* in, unsigned char* out,
                int width, int height){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(col < width && row < height) {
        int pixelValue = 0;
        int pixels = 0;
        
        for(int r = row - BLUR_SIZE; r < row + BLUR_SIZE + 1; r++) {
            for(int c = col - BLUR_SIZE; c < col + BLUR_SIZE + 1; c++) {
                if(c >=0 && c < width && r >= 0 && r < height) {
                    pixelValue += in[r * width + c];
                    pixels += 1;
                }
            }
        }
        out[row * width + col] = (unsigned char)(pixelValue / pixels);
    }
}

__global__
void matrixMulKernel(float* M, float* N, float* P, int width) {
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