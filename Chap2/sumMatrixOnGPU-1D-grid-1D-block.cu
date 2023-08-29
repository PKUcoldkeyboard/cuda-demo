#include <cuda_runtime.h>
#include <time.h>
#include "dbg.h"

clock_t hstart, hstop;
cudaEvent_t start, stop;

void checkResult(float *hostRef, float *gpuRef, const int N)
{
    double epsilon = 1e-8;
    bool match     = 1;

    for (int i = 0; i < N; i++)
    {
        if (abs(hostRef[i] - gpuRef[i]) > epsilon)
        {
            match = 0;
            log_err("Arrays do not match!");
            log_err("host %5.2f gpu %5.2f at current %d", hostRef[i], gpuRef[i], i);
            break;
        }
    }

    if (match)
    {
        log_info("Arrays match.");
    }
}

void init(float *A, float *B, const int N)
{
    for (int i = 0; i < N; i++)
    {
        A[i] = ((float)i) + 0.1335f;
        B[i] = 1.50f * ((float)i) + 0.9383f;
    }
}

void sumMatrixOnHost(float *A, float *B, float *C, const int nx, const int ny)
{
    float *ia = A;
    float *ib = B;
    float *ic = C;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
        {
            ic[ix] = ia[ix] + ib[ix];
        }
        ia += nx;
        ib += nx;
        ic += nx;
    }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;

    if (ix < nx)
    {
        for (int iy = 0; iy < ny; iy++)
        {
            int idx   = iy * nx + ix;
            MatC[idx] = MatA[idx] + MatB[idx];
        }
    }
}

int main(int argc, char **argv)
{
    log_info("[CUDA] %s Starting...", argv[0]);
    int dev = 0;
    cudaDeviceProp deviceProp;
    check_device(cudaGetDeviceProperties(&deviceProp, dev));
    log_info("Using Device %d: %s", dev, deviceProp.name);
    check_device(cudaSetDevice(dev));

    int nx     = 1 << 14;
    int ny     = 1 << 14;
    int nxy    = nx * ny;
    int nBytes = nxy * sizeof(float);
    log_info("Matrix size: nx %d ny %d", nx, ny);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nBytes);
    h_B     = (float *)malloc(nBytes);
    hostRef = (float *)malloc(nBytes);
    gpuRef  = (float *)malloc(nBytes);

    hstart = clock();
    init(h_A, h_B, nxy);
    hstop = clock();
    log_info("Init host data elapsed %f ms", (double)(hstop - hstart) / CLOCKS_PER_SEC * 1000.0);

    memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);
    hstart = clock();
    sumMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    hstop = clock();
    log_info("sumMatrixOnHost elapsed %f ms", (double)(hstop - hstart) / CLOCKS_PER_SEC * 1000.0);

    float *d_MatA, *d_MatB, *d_MatC;
    check_device(cudaMalloc((void **)&d_MatA, nBytes));
    check_device(cudaMalloc((void **)&d_MatB, nBytes));
    check_device(cudaMalloc((void **)&d_MatC, nBytes));

    check_device(cudaMemcpy(d_MatA, h_A, nBytes, cudaMemcpyHostToDevice));
    check_device(cudaMemcpy(d_MatB, h_B, nBytes, cudaMemcpyHostToDevice));

    int dimx = 32;
    dim3 block(dimx, 1);
    dim3 grid((nx + block.x - 1) / block.x, 1);

    check_device(cudaEventCreate(&start));
    check_device(cudaEventCreate(&stop));
    check_device(cudaEventRecord(start, 0));
    sumMatrixOnGPU2D<<<grid, block>>>(d_MatA, d_MatB, d_MatC, nx, ny);
    check_device(cudaEventRecord(stop, 0));
    check_device(cudaEventSynchronize(stop));
    float gpu_elapsed_time_ms;
    check_device(cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop));
    log_info("sumMatrixOnGPU2D <<<(%d,%d), (%d,%d)>>> elapsed %f ms", grid.x, grid.y, block.x,
             block.y, gpu_elapsed_time_ms);

    check_device(cudaGetLastError());
    check_device(cudaMemcpy(gpuRef, d_MatC, nBytes, cudaMemcpyDeviceToHost));
    checkResult(hostRef, gpuRef, nxy);
    check_device(cudaFree(d_MatA));
    check_device(cudaFree(d_MatB));
    check_device(cudaFree(d_MatC));
    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    check_device(cudaDeviceReset());
    return 0;
}