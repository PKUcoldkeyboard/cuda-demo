#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include "dbg.h"

void sumArrayOnHost(float *A, float* B, float* C, const uint32_t N) {
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

void initData(float* ip, int size, uint32_t seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0XFF) / 10.0f;
    }
}

__global__ void addKernel(float* d_A, float* d_B, float* d_C, const uint32_t N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        d_C[i] = d_A[i] + d_B[i];
    }
}

int main(int argc, char* argv[]) {
    log_info(">>> %s Starting...", argv[0]);

    int dev = 0;
    cudaDeviceProp deviceProp;
    check_device(cudaGetDeviceProperties(&deviceProp, dev));
    log_info(">>> Using device %d: %s", dev, deviceProp.name);

    uint32_t nElem = 1 << 24;
    log_info(">>> Vector Size: %d", nElem);
    size_t nBytes = nElem * sizeof(float);

    float *h_A, *h_B, *h_C, *gpuRef;
    h_A = (float *)malloc(nBytes);
    h_B = (float *)malloc(nBytes);
    h_C = (float *)malloc(nBytes);
    gpuRef = (float*)malloc(nBytes);

    initData(h_A, nElem, 1919180);
    initData(h_B, nElem, 114514);
    memset(h_C, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    sumArrayOnHost(h_A, h_B, h_C, nElem);
    
    float *d_A, *d_B, *d_C;

    check_device(cudaMalloc((void**)&d_A, nBytes));
    check_device(cudaMalloc((void**)&d_B, nBytes));
    check_device(cudaMalloc((void**)&d_C, nBytes));

    check_device(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    check_device(cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice));
    check_device(cudaMemcpy(d_C, gpuRef, nBytes, cudaMemcpyHostToDevice));

    int blockSize = atoi(argv[1]);

    dim3 block(blockSize);
    dim3 grid((nElem + block.x - 1) / block.x);
    log_info(">>> Executing configuration <<<%d, %d>>>", grid.x, block.x);

    cudaEvent_t start, stop;
    check_device(cudaEventCreate(&start));
    check_device(cudaEventCreate(&stop));
    check_device(cudaEventRecord(start, 0));
    addKernel<<<grid, block>>>(d_A, d_B, d_C, nElem);
    check_device(cudaEventRecord(stop, 0));
    check_device(cudaEventSynchronize(stop));
    check_device(cudaDeviceSynchronize());
    check_device(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    errno = 0;
    for (int i = 0; i < nElem; i++) {
        check(gpuRef[i] == h_C[i], "Arrays do not match! C[%d] = %.2f(%.2f)", i, gpuRef[i], h_C[i]);
    }
    log_info(">>> Arrays Match!");

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    log_info(">>> Elapsed time: %.3fs", elapsedTime);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaDeviceReset();

    free(h_A);
    free(h_B);
    free(h_C);
    free(gpuRef);
    return 0;
error:
    return 1;
}