#include <cuda_runtime.h>
#include <time.h>
#include "dbg.h"

int recursiveReduce(int *data, const int size)
{
    // 终止条件
    if (size == 1)
    {
        return data[0];
    }
    // 重新设置 stride 大小
    const int stride = size >> 1;

    // 遍历数组，将相邻 stride 的两个元素相加
    for (int i = 0; i < stride; i++)
    {
        data[i] += data[i + stride];
    }

    return recursiveReduce(data, stride);
}

// Kernel1: Neighbored Pair Implementation with divergence
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

// Kernel2: Interleaved Pair Implementation with less divergence（交错对实现）
__global__ void reduceInterleaved(int *g_idata, int *g_odata, unsigned int n)
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

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
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

// Kernel3: Neighbored Pair Implementation with less divergence
__global__ void reduceNeighboredLess(int *g_idata, int *g_odata, unsigned int n)
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
        int index = 2 * stride * tid;
        if (index < blockDim.x)
        {
            idata[index] += idata[index + stride];
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

// Kernel4: 循环展开，用一个线程块手动展开两个数据块的处理
__global__ void reduceUnrolling2(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程 id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // 循环展开
    if (idx + blockDim.x < n)
    {
        g_idata[idx] += g_idata[idx + blockDim.x];
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }
    // 写回全局内存
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// Kernel5: 循环展开，用一个线程块手动展开八个数据块的处理，并使用线程束内展开
__global__ void reduceUnrollWarp8(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程 id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // 循环展开
    if (idx + 7 * blockDim.x < n)
    {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        int b1       = g_idata[idx + 4 * blockDim.x];
        int b2       = g_idata[idx + 5 * blockDim.x];
        int b3       = g_idata[idx + 6 * blockDim.x];
        int b4       = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }
        __syncthreads();
    }

    // 使用线程束内展开
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }

    // 写回全局内存
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// Kernel6: 完全展开
__global__ void reduceCompleteUnrollWarp8(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程 id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // 循环展开
    if (idx + 7 * blockDim.x < n)
    {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        int b1       = g_idata[idx + 4 * blockDim.x];
        int b2       = g_idata[idx + 5 * blockDim.x];
        int b3       = g_idata[idx + 6 * blockDim.x];
        int b4       = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    if (blockDim.x >= 1024 && tid < 512)
    {
        idata[tid] += idata[tid + 512];
    }
    __syncthreads();

    if (blockDim.x >= 512 && tid < 256)
    {
        idata[tid] += idata[tid + 256];
    }

    __syncthreads();

    if (blockDim.x >= 256 && tid < 128)
    {
        idata[tid] += idata[tid + 128];
    }

    __syncthreads();

    if (blockDim.x >= 128 && tid < 64)
    {
        idata[tid] += idata[tid + 64];
    }

    __syncthreads();

    // 使用线程束内展开
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    // 写回全局内存
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// Kernel7: 模板函数
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnrollWarp(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程 id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 8 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 8;

    // 循环展开
    if (idx + 7 * blockDim.x < n)
    {
        int a1       = g_idata[idx];
        int a2       = g_idata[idx + blockDim.x];
        int a3       = g_idata[idx + 2 * blockDim.x];
        int a4       = g_idata[idx + 3 * blockDim.x];
        int b1       = g_idata[idx + 4 * blockDim.x];
        int b2       = g_idata[idx + 5 * blockDim.x];
        int b3       = g_idata[idx + 6 * blockDim.x];
        int b4       = g_idata[idx + 7 * blockDim.x];
        g_idata[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
    }
    __syncthreads();

    if (iBlockSize >= 1024 && tid < 512)
    {
        idata[tid] += idata[tid + 512];
    }
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)
    {
        idata[tid] += idata[tid + 256];
    }
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)
    {
        idata[tid] += idata[tid + 128];
    }
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)
    {
        idata[tid] += idata[tid + 64];
    }
    __syncthreads();

    // 使用线程束内展开
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    // 写回全局内存
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
    }
}

// Kernel8: 实现 Unrolling16
template <unsigned int iBlockSize>
__global__ void reduceCompleteUnrollWarp16(int *g_idata, int *g_odata, unsigned int n)
{
    // 设置线程 id
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 16 + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x * 16;

    // 循环展开
    int *ptr = g_idata + idx;
    int tmp  = 0;

    for (int i = 0; i < 16; i++)
    {
        tmp += *ptr;
        ptr += blockDim.x;
    }
    g_idata[idx] = tmp;
    __syncthreads();

    if (iBlockSize >= 1024 && tid < 512)
    {
        idata[tid] += idata[tid + 512];
    }
    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)
    {
        idata[tid] += idata[tid + 256];
    }
    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)
    {
        idata[tid] += idata[tid + 128];
    }
    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)
    {
        idata[tid] += idata[tid + 64];
    }
    __syncthreads();

    // 使用线程束内展开
    if (tid < 32)
    {
        volatile int *vmem = idata;
        vmem[tid] += vmem[tid + 32];
        vmem[tid] += vmem[tid + 16];
        vmem[tid] += vmem[tid + 8];
        vmem[tid] += vmem[tid + 4];
        vmem[tid] += vmem[tid + 2];
        vmem[tid] += vmem[tid + 1];
    }
    // 写回全局内存
    if (tid == 0)
    {
        g_odata[blockIdx.x] = idata[0];
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

    // 初始化
    int size = 1 << 24;
    log_info("With array size %d", size);

    // 执行配置
    int blockSize = 512;
    if (argc > 1)
    {
        blockSize = atoi(argv[1]);
    }
    dim3 block(blockSize, 1);
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
    int cpuSum     = recursiveReduce(tmp, size);
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

    // kernel 2: reduceNeighboredLess
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    reduceNeighboredLess<<<grid, block>>>(d_idata, d_odata, size);
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
        "gpu reduceNeighboredLess elapsed %f ms gpuSum: %d <<<grid %d block "
        "%d>>>",
        kernelTime, gpuSum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    reduceInterleaved<<<grid, block>>>(d_idata, d_odata, size);
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
        "gpu reduceInterleaved elapsed %f ms gpuSum: %d <<<grid %d block "
        "%d>>>",
        kernelTime, gpuSum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    reduceUnrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for (int i = 0; i < grid.x / 2; i++)
    {
        gpuSum += h_odata[i];
    }

    log_info(
        "gpu reduceUnrolling2 elapsed %f ms gpuSum: %d <<<grid %d block "
        "%d>>>",
        kernelTime, gpuSum, grid.x / 2, block.x);

    // kernel 5: reduceUnrollWarp8
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    reduceUnrollWarp8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for (int i = 0; i < grid.x / 8; i++)
    {
        gpuSum += h_odata[i];
    }

    log_info(
        "gpu reduceUnrollWarp8 elapsed %f ms gpuSum: %d <<<grid %d block "
        "%d>>>",
        kernelTime, gpuSum, grid.x / 8, block.x);

    // kernel 6: reduceCompleteUnrollWarp8
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    reduceCompleteUnrollWarp8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for (int i = 0; i < grid.x / 8; i++)
    {
        gpuSum += h_odata[i];
    }

    log_info(
        "gpu reduceCompleteUnrollWarp8 elapsed %f ms gpuSum: %d <<<grid %d "
        "block %d>>>",
        kernelTime, gpuSum, grid.x / 8, block.x);

    // kernel7: reduceCompleteUnrollWarp
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    switch (blockSize)
    {
        case 1024:
            reduceCompleteUnrollWarp<1024><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
        case 512:
            reduceCompleteUnrollWarp<512><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
        case 256:
            reduceCompleteUnrollWarp<256><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
        case 128:
            reduceCompleteUnrollWarp<128><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
        case 64:
            reduceCompleteUnrollWarp<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
            break;
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for (int i = 0; i < grid.x / 8; i++)
    {
        gpuSum += h_odata[i];
    }

    log_info(
        "gpu reduceCompleteUnrollWarp elapsed %f ms gpuSum: %d <<<grid %d "
        "block %d>>>",
        kernelTime, gpuSum, grid.x / 8, block.x);

    // kernel8: reduceCompleteUnrollWarp16
    CHECK(cudaMemcpy(d_idata, h_idata, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaEventRecord(start));
    switch (blockSize)
    {
        case 1024:
            reduceCompleteUnrollWarp16<1024><<<grid.x / 16, block>>>(d_idata, d_odata, size);
            break;
        case 512:
            reduceCompleteUnrollWarp16<512><<<grid.x / 16, block>>>(d_idata, d_odata, size);
            break;
        case 256:
            reduceCompleteUnrollWarp16<256><<<grid.x / 16, block>>>(d_idata, d_odata, size);
            break;
        case 128:
            reduceCompleteUnrollWarp16<128><<<grid.x / 16, block>>>(d_idata, d_odata, size);
            break;
        case 64:
            reduceCompleteUnrollWarp16<64><<<grid.x / 16, block>>>(d_idata, d_odata, size);
            break;
    }
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&kernelTime, start, stop));
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 16 * sizeof(int), cudaMemcpyDeviceToHost));
    gpuSum = 0;
    for (int i = 0; i < grid.x / 16; i++)
    {
        gpuSum += h_odata[i];
    }

    log_info(
        "gpu reduceCompleteUnrollWarp16 elapsed %f ms gpuSum: %d <<<grid %d "
        "block %d>>>",
        kernelTime, gpuSum, grid.x / 16, block.x);

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