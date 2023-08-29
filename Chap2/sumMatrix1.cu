#include <stdio.h>
#include <stdlib.h>
#include "dbg.h"

__global__ void sumMatrixOnGPU(float *matA, float *matB, float *matC, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        matC[idx] = matA[idx] + matB[idx];
    }
}

void initData(float* ip, int size, int seed) {
    srand(seed);
    for (int i = 0; i < size; i++) {
        ip[i] = (float) (rand() & 0XFF) / 10.0f;
    }
}

int main(int argc, char** argv) {
    log_info(">>> %s Starting...", argv[0]);
    
    int dev = 0;
    cudaDeviceProp deviceProp;
    check_device(cudaGetDeviceProperties(&deviceProp, dev));
    log_info(">>> Using device %d: %s", dev, deviceProp.name);

    int nx = 1 << 14;
    int ny = 1 << 14;
    log_info(">>> Matrix Size: (%d, %d)", nx, ny);

    int nBytes = nx * ny * sizeof(float);
    float *h_matA, *h_matB, *h_matC;
    h_matA = (float *)malloc(nBytes);
    h_matB = (float *)malloc(nBytes);
    h_matC = (float *)malloc(nBytes);

    memset(h_matA, 0, nBytes);
    memset(h_matB, 0, nBytes);

    initData(h_matA, nx * ny, 1919180);
    initData(h_matB, nx * ny, 114514);

    float *d_matA, *d_matB, *d_matC;
    cudaMalloc((void**)&d_matA, nBytes);
    cudaMalloc((void**)&d_matB, nBytes);
    cudaMalloc((void**)&d_matC, nBytes);
    
    cudaMemcpy(d_matA, h_matA, nBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, nBytes, cudaMemcpyHostToDevice);
    
    int dimx = 32;
    int dimy = 16;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    cudaEvent_t start, stop;
    check_device(cudaEventCreate(&start));
    check_device(cudaEventCreate(&stop));
    check_device(cudaEventRecord(start, 0));
    sumMatrixOnGPU<<<grid, block>>>(d_matA, d_matB, d_matC, nx, ny);
    check_device(cudaEventRecord(stop, 0));
    check_device(cudaEventSynchronize(stop));
    check_device(cudaDeviceSynchronize());
    check_device(cudaMemcpy(h_matC, d_matC, nBytes, cudaMemcpyDeviceToHost));

    float elapsedTime;
    check_device(cudaEventElapsedTime(&elapsedTime, start, stop));
    check_device(cudaEventDestroy(start));
    check_device(cudaEventDestroy(stop));
    log_info(">>> Elapsed time: %.3f ms", elapsedTime);
    
    cudaFree(d_matA);
    cudaFree(d_matB);
    cudaFree(d_matC);

    free(h_matA);
    free(h_matB);
    free(h_matC);

    cudaDeviceReset();
    
    return 0;
}