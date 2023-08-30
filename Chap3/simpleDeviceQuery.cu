#include <cuda_runtime.h>
#include "dbg.h"

/**
* 网格和线程块大小准则
* 1. 线程块大小应该是 32 的倍数
* 2. 避免块太小：每个块至少有 128 或 256 个线程
* 3. 根据内核资源的需求调整块大小
* 4. 块的数量要远远多于 SM 的数量，从而在设备中可以显示有足够的并行
* 5. 通过实验得到最佳执行配置和资源使用情况
*/
int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    log_info("Device %d: %s", dev, deviceProp.name);
    log_info("Number of SMs: %d", deviceProp.multiProcessorCount);
    log_info("Total amount of constant memory: %4.2f KB", deviceProp.totalConstMem / 1024.0);
    log_info("Total amount of shared memory per block: %4.2f KB",
             deviceProp.sharedMemPerBlock / 1024.0);
    log_info("Total number of registers available per block: %d", deviceProp.regsPerBlock);
    log_info("Warp size: %d", deviceProp.warpSize);
    log_info("Maximum number of threads per block: %d", deviceProp.maxThreadsPerBlock);
    log_info("Maximum number of threads per multiprocessor: %d",
             deviceProp.maxThreadsPerMultiProcessor);
    log_info("Maximum number of warps per multiprocessor: %d",
             deviceProp.maxThreadsPerMultiProcessor / 32);

    return 0;
}