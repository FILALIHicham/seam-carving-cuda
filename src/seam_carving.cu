#include "seam_carving.h"
#include "gpu_memory.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel to compute cumulative energy map
__global__ void compute_row_cumulative_energy(
    const float *energy,
    float *cumulative_energy,
    int *backtrack,
    int width,
    int height,
    int current_row
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int idx = current_row * width + x;
    
    if (current_row == 0) {
        // First row just copies the energy values
        cumulative_energy[idx] = energy[idx];
        backtrack[idx] = x;
        return;
    }

    // For other rows, find minimum from above
    float left = (x > 0) ? cumulative_energy[(current_row-1) * width + (x-1)] : INFINITY;
    float middle = cumulative_energy[(current_row-1) * width + x];
    float right = (x < width-1) ? cumulative_energy[(current_row-1) * width + (x+1)] : INFINITY;

    // Find minimum of the three possible paths
    float min_energy = middle;
    int min_x = x;

    if (left < min_energy) {
        min_energy = left;
        min_x = x - 1;
    }
    if (right < min_energy) {
        min_energy = right;
        min_x = x + 1;
    }

    // Store cumulative energy and backtrack pointer
    cumulative_energy[idx] = energy[idx] + min_energy;
    backtrack[idx] = min_x;
}

int* remove_seam_with_path(Image *img, float *device_energy) {
    int width = img->width;
    int height = img->height;

    size_t cumulative_size = width * height * sizeof(float);
    size_t backtrack_size = width * height * sizeof(int);

    // Allocate GPU memory
    float *cumulative_energy;
    int *backtrack;
    allocate_device_memory((void **)&cumulative_energy, cumulative_size);
    allocate_device_memory((void **)&backtrack, backtrack_size);

    // Compute cumulative energy row by row
    dim3 block_dim(256, 1);
    dim3 grid_dim((width + block_dim.x - 1) / block_dim.x, 1);

    for (int row = 0; row < height; row++) {
        compute_row_cumulative_energy<<<grid_dim, block_dim>>>(
            device_energy,
            cumulative_energy,
            backtrack,
            width,
            height,
            row
        );
        
        // Check for kernel errors after each row
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel error at row %d: %s\n", row, cudaGetErrorString(err));
            exit(1);
        }
        
        // Ensure row is complete before moving to next row
        cudaDeviceSynchronize();
    }

    // Copy results back to host
    float *host_cumulative = (float *)malloc(cumulative_size);
    int *host_backtrack = (int *)malloc(backtrack_size);
    if (!host_cumulative || !host_backtrack) {
        fprintf(stderr, "Failed to allocate host memory\n");
        exit(1);
    }
    
    copy_to_host(host_cumulative, cumulative_energy, cumulative_size);
    copy_to_host(host_backtrack, backtrack, backtrack_size);

    // Find minimum energy in bottom row
    float min_energy = host_cumulative[(height-1) * width];
    int seam_end = 0;
    for (int x = 1; x < width; x++) {
        float current = host_cumulative[(height-1) * width + x];
        if (current < min_energy) {
            min_energy = current;
            seam_end = x;
        }
    }

    // Backtrack to find seam path
    int *seam = (int *)malloc(height * sizeof(int));
    if (!seam) {
        fprintf(stderr, "Failed to allocate seam path memory\n");
        exit(1);
    }
    
    seam[height-1] = seam_end;
    for (int y = height-2; y >= 0; y--) {
        seam[y] = host_backtrack[(y+1) * width + seam[y+1]];
    }

    // Remove seam from image
    unsigned char *new_data = (unsigned char *)malloc((width-1) * height * 3);
    if (!new_data) {
        fprintf(stderr, "Failed to allocate new image memory\n");
        exit(1);
    }

    for (int y = 0; y < height; y++) {
        int offset = 0;
        for (int x = 0; x < width; x++) {
            if (x == seam[y]) {
                offset = -1;
                continue;
            }
            for (int c = 0; c < 3; c++) {
                new_data[(y * (width-1) + (x + offset)) * 3 + c] = 
                    img->data[(y * width + x) * 3 + c];
            }
        }
    }

    // Update image
    free(img->data);
    img->data = new_data;
    img->width--;

    // Cleanup
    free(host_cumulative);
    free(host_backtrack);
    free_device_memory(cumulative_energy);
    free_device_memory(backtrack);

    return seam;
}