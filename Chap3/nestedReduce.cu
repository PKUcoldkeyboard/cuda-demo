#include <cuda_runtime.h>
#include <time.h>
#include "dbg.h"

int cpuRecursiveReduce(int *data, const int size)
{
    if (size == 1)
    {
        return data[0];
    }

    const int stride = size / 2;

    for (int i = 0; i < stride; ++i)
    {
        data[i] += data[i + stride];
    }

    return cpuRecursiveReduce(data, stride);
}

__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程 id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 将全局内存数据加载到共享内存中
    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n)
    {
        return;
    }

    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if ((tid % (2 * stride)) == 0)
        {
            idata[tid] += idata[tid + stride];
        }
        // 等待所有线程完成本次迭代
        __syncthreads();
    }
    // 写回全局内存
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

__global__ void gpuRecursiveReduce(int *g_idata, int *g_odata, unsigned isize)
{
    // 设置线程 id
    unsigned int tid = threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;
    int *odata = &g_odata[blockIdx.x];

    // 边界条件
    if (isize == 2 && tid == 0)
    {
        g_odata[blockIdx.x] = idata[0] + idata[1];
        return;
    }

    int istride = isize >> 1;

    if (istride > 1 && tid < istride)
    {
        idata[tid] += idata[tid + istride];
        
        if (tid == 0)
        {
            gpuRecursiveReduce<<<1, istride>>>(idata, odata, istride);
        }
    }
}

int main(int argc, char **argv)
{
    // 设置当前使用的GPU设备
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    log_info("Using Device %d: %s", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // 设置执行配置
    int nBlock   = 2048;
    int nThreads = 512;

    if (argc > 1)
    {
        nBlock = atoi(argv[1]);
    }

    if (argc > 2)
    {
        nThreads = atoi(argv[2]);
    }

    int size = nBlock * nThreads;

    dim3 block(nThreads, 1);
    dim3 grid((size + block.x - 1) / block.x, 1);
    log_info("Execution Configure (block %d grid %d)", block.x, grid.x);

    // 分配主机内存
    size_t nBytes = size * sizeof(int);
    int *h_idata  = (int *)malloc(nBytes);
    int *h_odata  = (int *)malloc(grid.x * sizeof(int));
    int *tmp      = (int *)malloc(nBytes);

    // 初始化数组
    for (int i = 0; i < size; i++)
    {
        // 使用掩码 0xFF 限制随机数的范围至 0-255
        h_idata[i] = (int)(rand() & 0xFF);
    }

    memcpy(tmp, h_idata, nBytes);

    // 分配设备内存
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **)&d_idata, nBytes));
    CHECK(cudaMalloc((void **)&d_odata, grid.x * sizeof(int)));

    // cpu reduction
    clock_t iStart = clock();
    int cpuSum     = cpuRecursiveReduce(tmp, size);
    clock_t iElaps = clock() - iStart;
    log_info("cpu reduce elapsed %f ms cpuSum: %d", (double)iElaps / CLOCKS_PER_SEC * 1000.0,
             cpuSum);

    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventRecord(start));
    reduceNeighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float kernelTime;
    CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    int gpuSum = 0;
    for (int i = 0; i < grid.x; i++)
    {
        gpuSum += h_odata[i];
    }

    log_info(
        "gpu reduceNeighbored elapsed %f ms gpuSum: %d <<<grid %d block "
        "%d>>>",
        kernelTime, gpuSum, grid.x, block.x);

    // kernel2: gpuRecursiveReduce
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    gpuRecursiveReduce<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for (int i = 0; i < grid.x; i++)
    {
        gpuSum += h_odata[i];
    }

    log_info(
        "gpu gpuRecursiveReduce elapsed %f ms gpuSum: %d <<<grid %d block "
        "%d>>>",
        kernelTime, gpuSum, grid.x, block.x);

    // 释放资源
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));
    free(h_idata);
    free(h_odata);
    free(tmp);
    return 0;
}