#include <cuda_runtime.h>
#include <stdio.h>
#include "dbg.h"

int main(int argc, char** argv) {
    log_info(" <<< %s Starting...", argv[0]);
    int deviceCount = 0;
    check_device(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        log_err("There are no available devices that support CUDA");
    } else {
        log_info("<<< Detected %d CUDA Capable devices", deviceCount);
    }

    int dev = 0, driverVersion = 0, runtimeVersion = 0;
    check_device(cudaSetDevice(dev));
    cudaDeviceProp deviceProp;
    check_device(cudaGetDeviceProperties(&deviceProp, dev));
    log_info("<<< Device %d: %s", dev, deviceProp.name);
    check_device(cudaDriverGetVersion(&driverVersion));
    check_device(cudaRuntimeGetVersion(&runtimeVersion));
    
    log_info("<<< CUDA Driver version: %d.%d", driverVersion / 1000, (driverVersion % 100) / 10);
    log_info("<<< CUDA Runtime version: %d.%d", runtimeVersion / 1000, (runtimeVersion % 100) / 10);
    log_info("<<< CUDA Capability Major/Minor version: %d.%d", deviceProp.major, deviceProp.minor);
    log_info("<<< Total amount of global memory: %.2f Mbytes", deviceProp.totalGlobalMem / pow(1024.0, 3));
    log_info("<<< GPU Clock rate: %0.2f GHz", deviceProp.clockRate * 1e-6f);
    log_info("<<< Memory Clock rate: %.0f MHz", deviceProp.memoryClockRate * 1e-3f);
    log_info("<<< Memory Bus Width: %d-bit", deviceProp.memoryBusWidth);
    
    if (deviceProp.l2CacheSize) {
        log_info("<<< L2 Cache size: %d bytes", deviceProp.l2CacheSize);
    }

    log_info("<<< Max Texture dimension size (x, y, z) 1D=(%d), 2D=(%d, %d), 3D=(%d, %d, %d)",
            deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
            deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
    log_info("<<< Max Layered Texture size (dim) x layers 1D=(%d) x %d, 2D=(%d, %d) x %d",
            deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
            deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
            deviceProp.maxTexture2DLayered[2]);
    log_info("<<< Total amount of constant memory: %.2f Mbytes", deviceProp.totalConstMem / pow(1024.0, 3));
    log_info("<<< Total amount of shared memory per block: %.2f Mbytes", deviceProp.sharedMemPerBlock / pow(1024.0, 3));
    log_info("<<< Total amount of registers avalible per block: %d", deviceProp.regsPerBlock);
    log_info("<<< Warp size: %d", deviceProp.warpSize);
    log_info("<<< Max number of threads per multiprocessor: %d", deviceProp.maxThreadsPerMultiProcessor);
    log_info("<<< Max size of each dimension of a block: %d x %d x %d", deviceProp.maxThreadsDim[0],
            deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    log_info("<<< Max size of each dimension of a grid: %d x %d x %d", deviceProp.maxGridSize[0],
            deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    log_info("<<< Max memory pitch: %lu bytes", deviceProp.memPitch);
}