#include "seam_carving.h"
#include "gpu_memory.h"
#include "energy.h"
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

// Function to store k seams before insertion
int** find_k_seams(Image *img, float *device_energy, int k, int direction, int *seam_lengths) {
    int width = img->width;
    int height = img->height;
    
    // Allocate array to store k seams
    int **seams = (int **)malloc(k * sizeof(int *));
    if (!seams) {
        fprintf(stderr, "Failed to allocate seams array\n");
        exit(1);
    }
    
    // Create a temporary image for seam calculation
    Image temp_img = {width, height, NULL};
    temp_img.data = (unsigned char *)malloc(width * height * 3);
    if (!temp_img.data) {
        fprintf(stderr, "Failed to allocate temporary image data\n");
        exit(1);
    }
    memcpy(temp_img.data, img->data, width * height * 3);
    
    // Calculate initial energy size
    size_t energy_size = width * height * sizeof(float);
    
    // Create temporary energy map
    float *temp_energy;
    allocate_device_memory((void **)&temp_energy, energy_size);
    
    // Copy initial energy map
    copy_to_device(device_energy, temp_energy, energy_size);
    
    // Find k seams
    for (int i = 0; i < k; i++) {
        // Compute energy for current state
        compute_energy(&temp_img, temp_energy);
        
        // Find seam
        seam_lengths[i] = direction == 0 ? height : width;
        seams[i] = remove_seam_with_path(&temp_img, temp_energy, direction);
        
        if (!seams[i]) {
            fprintf(stderr, "Failed to find seam %d\n", i);
            exit(1);
        }
        
        // Update energy map size after removal
        if (direction == 0) {
            width--;
            energy_size = width * height * sizeof(float);
        } else {
            height--;
            energy_size = width * height * sizeof(float);
        }
        
        // Reallocate temp_energy with new size
        float *new_temp_energy;
        allocate_device_memory((void **)&new_temp_energy, energy_size);
        free_device_memory(temp_energy);
        temp_energy = new_temp_energy;
        
        // Skip energy update for the last iteration
        if (i < k - 1) {
            compute_energy(&temp_img, temp_energy);
        }
    }
    
    // Cleanup temporary resources
    free_image(temp_img);
    free_device_memory(temp_energy);
    
    return seams;
}

// Function to insert a single pre-calculated seam
void insert_seam(Image *img, int *seam, int direction) {
    int width = img->width;
    int height = img->height;
    unsigned char *new_data = NULL;
    
    if (direction == 0) {  // Vertical seam
        new_data = (unsigned char *)malloc((width + 1) * height * 3);
        for (int y = 0; y < height; y++) {
            int offset = 0;
            for (int x = 0; x < width + 1; x++) {
                if (x == seam[y]) {
                    // Average with neighbors
                    int left_x = x - 1;
                    int right_x = x;
                    if (left_x < 0) left_x = 0;
                    if (right_x >= width) right_x = width - 1;
                    
                    for (int c = 0; c < 3; c++) {
                        int left_val = img->data[(y * width + left_x) * 3 + c];
                        int right_val = img->data[(y * width + right_x) * 3 + c];
                        new_data[(y * (width + 1) + x) * 3 + c] = (left_val + right_val) / 2;
                    }
                    offset = 1;
                } else {
                    for (int c = 0; c < 3; c++) {
                        new_data[(y * (width + 1) + x) * 3 + c] = 
                            img->data[(y * width + (x - offset)) * 3 + c];
                    }
                }
            }
        }
        img->width++;
    } else {  // Horizontal seam
        new_data = (unsigned char *)malloc(width * (height + 1) * 3);
        for (int y = 0; y < height + 1; y++) {
            if (y == seam[0]) {
                // Average with neighbors
                int above_y = y - 1;
                int below_y = y;
                if (above_y < 0) above_y = 0;
                if (below_y >= height) below_y = height - 1;
                
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < 3; c++) {
                        int above_val = img->data[(above_y * width + x) * 3 + c];
                        int below_val = img->data[(below_y * width + x) * 3 + c];
                        new_data[(y * width + x) * 3 + c] = (above_val + below_val) / 2;
                    }
                }
            } else {
                int src_y = y > seam[0] ? y - 1 : y;
                for (int x = 0; x < width; x++) {
                    for (int c = 0; c < 3; c++) {
                        new_data[(y * width + x) * 3 + c] = 
                            img->data[(src_y * width + x) * 3 + c];
                    }
                }
            }
        }
        img->height++;
    }
    
    // Update image data
    free(img->data);
    img->data = new_data;
}

