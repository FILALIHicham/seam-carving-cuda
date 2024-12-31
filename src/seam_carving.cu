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
    int current_row,
    int direction
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= width) return;

    int idx, energy_idx;
    if (direction == 0) {  // Vertical seam
        idx = current_row * width + x;
        energy_idx = idx;
    } else {  // Horizontal seam
        idx = current_row + x * height;  // Transposed layout
        energy_idx = current_row * width + x;  // Original layout for energy
    }

    if (current_row == 0) {
        // First row just copies the energy values
        cumulative_energy[idx] = energy[energy_idx];
        backtrack[idx] = x;
        return;
    }

    // Neighbor calculations
    float left, middle, right;
    if (direction == 0) {  // Vertical seam
        int prev_row = (current_row - 1) * width;
        left = (x > 0) ? cumulative_energy[prev_row + (x - 1)] : INFINITY;
        middle = cumulative_energy[prev_row + x];
        right = (x < width - 1) ? cumulative_energy[prev_row + (x + 1)] : INFINITY;
    } else {  // Horizontal seam
        int prev_row = current_row - 1;
        left = (x > 0) ? cumulative_energy[prev_row + (x - 1) * height] : INFINITY;
        middle = cumulative_energy[prev_row + x * height];
        right = (x < width - 1) ? cumulative_energy[prev_row + (x + 1) * height] : INFINITY;
    }

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
    cumulative_energy[idx] = energy[energy_idx] + min_energy;
    backtrack[idx] = min_x;
}

int* remove_seam_with_path(Image *img, float *device_energy, int direction) {
    int width = img->width;
    int height = img->height;
    int seam_length = direction == 0 ? height : width;
    int search_width = direction == 0 ? width : height;

    size_t cumulative_size = width * height * sizeof(float);
    size_t backtrack_size = width * height * sizeof(int);

    // Allocate GPU memory
    float *cumulative_energy;
    int *backtrack;
    allocate_device_memory((void **)&cumulative_energy, cumulative_size);
    allocate_device_memory((void **)&backtrack, backtrack_size);

    // Compute cumulative energy row by row
    dim3 block_dim(256, 1);
    dim3 grid_dim((search_width + block_dim.x - 1) / block_dim.x, 1);

    for (int row = 0; row < seam_length; row++) {
        compute_row_cumulative_energy<<<grid_dim, block_dim>>>(
            device_energy,
            cumulative_energy,
            backtrack,
            search_width,
            seam_length,
            row,
            direction
        );
        
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel error at row %d: %s\n", row, cudaGetErrorString(err));
            exit(1);
        }
        
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

    // Find minimum energy in last row/column
    float min_energy = INFINITY;
    int seam_end = 0;
    int last_row_offset = (seam_length - 1) * (direction == 0 ? search_width : 1);

    for (int x = 0; x < search_width; x++) {
        float current = direction == 0 ?
            host_cumulative[last_row_offset + x] :
            host_cumulative[x * seam_length + (seam_length - 1)];
        
        if (current < min_energy) {
            min_energy = current;
            seam_end = x;
        }
    }

    // Backtrack to find seam path
    int *seam = (int *)malloc(seam_length * sizeof(int));
    if (!seam) {
        fprintf(stderr, "Failed to allocate seam path memory\n");
        exit(1);
    }
    
    seam[seam_length-1] = seam_end;

    for (int i = seam_length - 2; i >= 0; i--) {
        int curr_x = seam[i + 1];
        if (direction == 0) {
            seam[i] = host_backtrack[i * search_width + curr_x];
        } else {
            seam[i] = host_backtrack[curr_x * seam_length + i];
        }
    }

    // Remove seam from image
    unsigned char *new_data = NULL;

    if (direction == 0) {  // Vertical seam -> new width = width - 1
        new_data = (unsigned char *)malloc((width - 1) * height * 3);
    } else {               // Horizontal seam -> new height = height - 1
        new_data = (unsigned char *)malloc(width * (height - 1) * 3);
    }

    if (direction == 0) {  // Vertical seam
        for (int y = 0; y < height; y++) {
            int offset = 0;
            for (int x = 0; x < width; x++) {
                if (x == seam[y]) {
                    offset = -1;
                    continue;
                }
                for (int c = 0; c < 3; c++) {
                    new_data[(y * (width - 1) + (x + offset)) * 3 + c] = 
                        img->data[(y * width + x) * 3 + c];
                }
            }
        }
        img->width--;
    } else {  // Horizontal seam
        for (int y = 0; y < height; y++) {
            if (y == seam[0]) continue;
            
            int new_y = y > seam[0] ? y - 1 : y;
            for (int x = 0; x < width; x++) {
                for (int c = 0; c < 3; c++) {
                    new_data[(new_y * width + x) * 3 + c] = 
                        img->data[(y * width + x) * 3 + c];
                }
            }
        }
        img->height--;
    }

    // Update image
    free(img->data);
    img->data = new_data;

    // Cleanup
    free(host_cumulative);
    free(host_backtrack);
    free_device_memory(cumulative_energy);
    free_device_memory(backtrack);

    return seam;
}