#include <cuda_runtime.h>
#include "dbg.h"

void printMatrix(int *C, const int nx, const int ny)
{
    int *ic = C;
    log_info("Matrix: (%d.%d)", nx, ny);

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            printf("%3d", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
}

__global__ void printThreadIndexKernel(int *A, const int nx, const int ny)
{
    int ix           = threadIdx.x + blockIdx.x * blockDim.x;
    int iy           = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    printf(
        "thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) "
        "global index %2d ival %2d\n",
        threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx, A[idx]);
}

int main(int argc, char **argv)
{
    log_info("[CUDA] %s Starting...", argv[0]);
    int dev = 0;
    cudaDeviceProp deviceProp;
    check_device(cudaGetDeviceProperties(&deviceProp, dev));
    log_info("Using Device %d: %s", dev, deviceProp.name);
    check_device(cudaSetDevice(dev));

    int nx     = 8;
    int ny     = 6;
    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);

    int *h_A = (int *)malloc(nBytes);

    for (int i = 0; i < nxy; i++)
    {
        h_A[i] = i;
    }

    printMatrix(h_A, nx, ny);

    int *d_MatA;
    check_device(cudaMalloc((void **)&d_MatA, nBytes));
    check_device(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));

    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    printThreadIndexKernel<<<grid, block>>>(d_MatA, nx, ny);
    check_device(cudaGetLastError());

    check_device(cudaFree(d_MatA));
    free(h_A);

    check_device(cudaDeviceReset());

    return 0;
}