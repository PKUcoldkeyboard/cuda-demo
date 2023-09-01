NVCC = nvcc
NVCC_FLAGS = -O3

all: hello

hello: hello.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -f hello