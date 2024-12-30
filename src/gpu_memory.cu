#include "gpu_memory.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Allocates memory on the GPU
void allocate_device_memory(void **dev_ptr, size_t size) {
    cudaError_t err = cudaMalloc(dev_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMalloc failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE); // Exit if memory allocation fails
    }
}

// Frees memory on the GPU
void free_device_memory(void *dev_ptr) {
    cudaError_t err = cudaFree(dev_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaFree failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE); 
    }
}

// Copies data from host (CPU) to device (GPU)
void copy_to_device(void *dst, const void *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMemcpy (HostToDevice) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// Copies data from device (GPU) to host (CPU)
void copy_to_host(void *dst, const void *src, size_t size) {
    cudaError_t err = cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: cudaMemcpy (DeviceToHost) failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE); 
    }
}
