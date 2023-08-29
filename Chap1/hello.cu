#include "dbg.h"

/**
 *  一个简单的 CUDA 程序
 *  该程序会打印 "Hello World from CPU" 一次，打印 "Hello World from GPU" 10 次 （10 个 CUDA 线程）
 */

__global__ void helloFromGPU(void)
{
    int threadId = threadIdx.x;
    printf("Hello World from GPU! threadId = %d\n", threadId);
}

void helloFromCPU(void)
{
    printf("Hello World from CPU!\n");
}

int main(int argc, char **argv)
{
    helloFromCPU();
    int blockDim = 10;
    int gridDim  = 1;
    helloFromGPU<<<gridDim, blockDim>>>();
    check_device(cudaDeviceReset());
    return 0;
}