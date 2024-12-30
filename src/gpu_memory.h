#ifndef GPU_MEMORY_H
#define GPU_MEMORY_H

#include <stddef.h> // For size_t

// Allocates memory on the GPU
void allocate_device_memory(void **dev_ptr, size_t size);

// Frees memory on the GPU
void free_device_memory(void *dev_ptr);

// Copies data from the host (CPU) to the device (GPU)
void copy_to_device(void *dst, const void *src, size_t size);

// Copies data from the device (GPU) to the host (CPU)
void copy_to_host(void *dst, const void *src, size_t size);

#endif // GPU_MEMORY_H
