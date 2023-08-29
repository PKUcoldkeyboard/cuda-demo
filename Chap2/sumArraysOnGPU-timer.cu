#include <cuda_runtime.h>
#include <time.h>
#include "dbg.h"

cudaEvent_t start, stop;
clock_t hstart, hstop;

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

__global__ void sumArraysOnGPUKernel(float *A, float *B, float *C, const int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

void sumArraysOnGPU(float *A, float *B, float *C, const int N)
{
    int size = N * sizeof(float);
    float *d_A, *d_B, *d_C;

    check_device(cudaMalloc((void **)&d_A, size));
    check_device(cudaMalloc((void **)&d_B, size));
    check_device(cudaMalloc((void **)&d_C, size));

    check_device(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    check_device(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 512;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    check_device(cudaEventCreate(&start));
    check_device(cudaEventCreate(&stop));
    check_device(cudaEventRecord(start, 0));
    sumArraysOnGPUKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    check_device(cudaEventRecord(stop, 0));
    check_device(cudaEventSynchronize(stop));
    check_device(cudaDeviceSynchronize());

    check_device(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

    check_device(cudaFree(d_A));
    check_device(cudaFree(d_B));
    check_device(cudaFree(d_C));
}

void sumArraysOnHost(float *A, float *B, float *C, const int N)
{
    hstart = clock();
    for (int i = 0; i < N; i++)
    {
        C[i] = A[i] + B[i];
    }
    hstop = clock();
}

int main(int argc, char **argv)
{
    log_info("[CUDA] sumArraysOnGPU...");
    int dev = 0;
    cudaDeviceProp deviceProp;
    check_device(cudaGetDeviceProperties(&deviceProp, dev));
    log_info("Using Device %d: %s", dev, deviceProp.name);
    check_device(cudaSetDevice(dev));

    int nElem = 1 << 24;
    log_info("Vector size %d", nElem);

    float *h_A, *h_B, *hostRef, *gpuRef;
    h_A     = (float *)malloc(nElem * sizeof(float));
    h_B     = (float *)malloc(nElem * sizeof(float));
    hostRef = (float *)malloc(nElem * sizeof(float));
    gpuRef  = (float *)malloc(nElem * sizeof(float));

    init(h_A, h_B, nElem);
    sumArraysOnHost(h_A, h_B, hostRef, nElem);
    sumArraysOnGPU(h_A, h_B, gpuRef, nElem);

    checkResult(hostRef, gpuRef, nElem);

    float gpuTime = 0.0f;
    check_device(cudaEventElapsedTime(&gpuTime, start, stop));
    log_info("sumArraysOnGPU elapsed %f ms", gpuTime);

    log_info("sumArraysOnHost time elapsed %f ms",
             (double)(hstop - hstart) / CLOCKS_PER_SEC * 1000.0);

    free(h_A);
    free(h_B);
    free(hostRef);
    free(gpuRef);
    return 0;
}