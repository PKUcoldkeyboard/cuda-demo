#include <cuda_runtime.h>
#include "dbg.h"

/**
* This example demonstrates the use of recursion in CUDA
* The kernel launches itself recursively until the value of iSize is 1
*/
__global__ void nestedHelloWorld(const int iSize, int iDepth) {
    int tid = threadIdx.x;
    printf("Recursion=%d: Hello World from thread %d block %d\n", iDepth, tid, blockIdx.x);

    if (iSize == 1) {
        return;
    }

    int nThreads = iSize >> 1;

    if (tid == 0 && nThreads > 0) {
        nestedHelloWorld<<<1, nThreads>>>(nThreads, ++iDepth);
        printf("------> nested execution depth: %d\n", iDepth);
    }
}

int main(int argc, char **argv) {
    int size = 8;
    int blockSize = 8;
    int igrid = 1;

    if (argc > 1) {
        igrid = atoi(argv[1]);
        size = igrid * blockSize;
    }

    dim3 block(blockSize, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    log_info("%s Execution Configuration: grid %d block %d", argv[0], grid.x, block.x);

    nestedHelloWorld<<<grid, block>>>(block.x, 0);

    CHECK(cudaGetLastError());
    CHECK(cudaDeviceReset());
    return 0;
}