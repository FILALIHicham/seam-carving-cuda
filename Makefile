NVCC = nvcc
CFLAGS = -O2 -diag-suppress=550

all: seam_carving

seam_carving: src/main.cu src/utils.cu src/gpu_memory.cu src/energy.cu src/seam_carving.cu
	$(NVCC) $(CFLAGS) -o build/seam_carving src/main.cu src/utils.cu src/gpu_memory.cu src/energy.cu src/seam_carving.cu -Iinclude

clean:
	rm -rf build/seam_carving
