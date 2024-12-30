#include "energy.h"
#include "gpu_memory.h"
#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel for energy computation
__global__ void compute_energy_kernel(const unsigned char *input, float *energy, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Ensure thread is within bounds
    if (x >= width || y >= height) return;

    int idx = y * width + x;

    // Sobel filter kernels
    const int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    float grad_x = 0.0f, grad_y = 0.0f;

    // Apply Sobel filter
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);

            int neighbor_idx = ny * width + nx;

            // Use grayscale intensity (average of RGB channels)
            unsigned char pixel_value = (input[neighbor_idx * 3] +
                                         input[neighbor_idx * 3 + 1] +
                                         input[neighbor_idx * 3 + 2]) / 3;

            grad_x += gx[dy + 1][dx + 1] * pixel_value;
            grad_y += gy[dy + 1][dx + 1] * pixel_value;
        }
    }

    energy[idx] = sqrtf(grad_x * grad_x + grad_y * grad_y);
}


// Host function to compute energy
void compute_energy(const Image *input, float *device_energy) {
    int width = input->width;
    int height = input->height;
    size_t img_size = width * height * 3 * sizeof(unsigned char);

    // Allocate device memory for input image
    unsigned char *device_input;
    allocate_device_memory((void **)&device_input, img_size);

    // Copy image data to device
    copy_to_device(device_input, input->data, img_size);

    // Launch kernel
    dim3 block_dim(16, 16);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 
                  (height + block_dim.y - 1) / block_dim.y);
    
    compute_energy_kernel<<<grid_dim, block_dim>>>(device_input, device_energy, width, height);
    
    // Check for kernel execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Kernel execution failed: %s\n", cudaGetErrorString(err));
        free_device_memory(device_input);
        exit(EXIT_FAILURE);
    }

    // Synchronize to ensure kernel completion
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Error: Device synchronization failed: %s\n", cudaGetErrorString(err));
        free_device_memory(device_input);
        exit(EXIT_FAILURE);
    }

    free_device_memory(device_input);
}
