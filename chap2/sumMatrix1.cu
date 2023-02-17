#include <stdio.h>
#include <stdlib.h>
#include "dbg.h"

__global__ void sumMatrixOnGPU(float *matA, float *matB, float *matC, int nx, int ny) {
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iy = blockIdx.y * blockDim.y + threadIdy.y;
    int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {
        matC[idx] = matA[idx] + matB[idx];
    }
}

int main(int argc, char** argv) {
    
}