#include <stdio.h>
#include <stdlib.h>
#include "dbg.h"

__global__ void helloFromGPU(void)
{
    int threadId = threadIdx.x;
    printf("Hello from GPU! threadId: %d\n", threadId);
}

int main(int argc, char **argv)
{
    cudaError_t cudaStatus = cudaSuccess;
    printf("Hello from CPU!\n");

    int blockDim = 10;
    int gridDim  = 1;
    helloFromGPU<<<gridDim, blockDim>>>();
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess)
    {
        log_err("Failed to synchronize, %s\n", cudaGetErrorString(cudaStatus));
        cudaDeviceReset();
        return -1;
    }
    cudaDeviceReset();
    return 0;
}