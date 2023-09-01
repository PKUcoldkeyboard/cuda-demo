NVCC = nvcc
NVCC_FLAGS = -O3 -lcudadevrt --relocatable-device-code true

CU_APPS = simpleDivergence simpleDeviceQuery nestedHelloWorld reduceInteger \
		  nestedReduce

all: $(CU_APPS) $(C_APPS)

%: %.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f $(CU_APPS) $(C_APPS)