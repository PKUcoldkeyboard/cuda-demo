NVCC = nvcc
NVCC_FLAGS = -O3
CC = gcc
CC_FLAGS = -O3

CU_APPS = checkDeviceInfo checkDimension defineGridBlock sumArraysOnGPU-timer \
		  checkThreadIndex sumMatrixOnGPU-2D-grid-2D-block sumMatrixOnGPU-2D-grid-1D-block \
		  sumMatrixOnGPU-1D-grid-1D-block
C_APPS = sumArraysOnHost

all: $(CU_APPS) $(C_APPS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

%: %.c
	$(CC) $(CC_FLAGS) $< -o $@

clean:
	rm -f $(CU_APPS) $(C_APPS)