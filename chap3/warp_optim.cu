#include <stdio.h>
#include <stdlib.h>
#include "dbg.h"


__global__ void warmingup(float* c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if (index % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[index] = a + b;
}

__global__ void mathKernel1(float *c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    bool ipred = (index % 2 == 0);
    if (ipred) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[index] = a + b;
}

__global__ void mathKernel2(float* c) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;

    if ((index / warpSize) % 2 == 0) {
        a = 100.0f;
    } else {
        b = 200.0f;
    }
    c[index] = a + b;
}


int main(int argc, char** argv) {
    log_info("<<< %s Starting...", argv[0]);
    // set up device
    int device = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    cudaSetDevice(device);

    log_info("<<< Using device: %d: %s", device, deviceProp.name);

    // set up data size
    int size = 64;
    int blockSize = 64;
    if (argc <= 1) {
        log_err("<<< Usage: %s <blockSize> <dataSize>", argv[0]);
    }
    if (argc > 1) {
        blockSize = atoi(argv[1]);
    }
    if (argc > 2) {
        size = atoi(argv[2]);
    }

    // set up execution configuration
    dim3 block(blockSize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    log_info("<<< Execution Configuration: (block %d grid %d)", block.x, grid.x);

    // allocate gpu memory
    float* d_C;
    cudaMalloc((void**)&d_C, size * sizeof(float));

    // run a warmup kernel to remove overhead
    cudaEvent_t start, stop;
    float elapsedTime;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();
    cudaEventRecord(start, 0);
    warmingup<<<grid,block>>>(d_C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    log_info("<<< warmup elapsed time: %.3f ms", elapsedTime);

    // run kernel1
    cudaEventRecord(start, 0);
    mathKernel1<<<grid,block>>>(d_C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    log_info("<<< kernel1 elapsed time: %.3f ms", elapsedTime);

    // run kernel2
    cudaEventRecord(start, 0);
    mathKernel2<<<grid,block>>>(d_C);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsedTime, start, stop);
    log_info("<<< kernel2 elapsed time: %.3f ms", elapsedTime);

    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}