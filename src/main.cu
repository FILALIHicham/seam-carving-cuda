#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "gpu_memory.h"
#include "energy.h"
#include "seam_carving.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s -i <input_image> -o <output_image> [-n <number_of_seams>] [--insert] [--save-seams] [--horizontal]\n", argv[0]);
        return 1;
    }
    
    // Default values for options
    int num_seams = 1;
    int save_seam_path = 0;
    int direction = 0; // 0 for vertical, 1 for horizontal
    int insert_mode = 0; // 0 for removal, 1 for insertion
    char *input_image_path = NULL;
    char *output_image_path = NULL;
    
    // Parse CLI arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_image_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_image_path = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_seams = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--save-seams") == 0) {
            save_seam_path = 1;
        } else if (strcmp(argv[i], "--horizontal") == 0) {
            direction = 1;
        } else if (strcmp(argv[i], "--insert") == 0) {
            insert_mode = 1;
        }
    }
    
    // Validate required arguments
    if (input_image_path == NULL || output_image_path == NULL) {
        printf("Error: Both -i <input_image> and -o <output_image> options are required.\n");
        return 1;
    }
    
    // Load the input image
    Image img = load_image(input_image_path);
    printf("Image loaded: %s (%dx%d)\n", input_image_path, img.width, img.height);
    
    // Allocate GPU memory for the energy map
    float *device_energy;
    allocate_device_memory((void **)&device_energy, img.width * img.height * sizeof(float));
    
    if (insert_mode) {
        // For insertion, first find all k seams
        int *seam_lengths = (int *)malloc(num_seams * sizeof(int));
        int **seams = find_k_seams(&img, device_energy, num_seams, direction, seam_lengths);
        
        // Then insert all seams
        for (int i = 0; i < num_seams; i++) {
            insert_seam(&img, seams[i], direction);
            printf("%s seam %d inserted.\n", direction == 0 ? "Vertical" : "Horizontal", i + 1);
            
            // Save seam path if enabled
            if (save_seam_path) {
                unsigned char *overlay_image = (unsigned char *)malloc(img.width * img.height * 3);
                memcpy(overlay_image, img.data, img.width * img.height * 3);
                
                if (direction == 0) {  // Vertical seam
                    for (int y = 0; y < img.height; y++) {
                        int seam_x = seams[i][y];
                        int idx = (y * img.width + seam_x) * 3;
                        overlay_image[idx] = 255;     // Red
                        overlay_image[idx + 1] = 0;   // Green
                        overlay_image[idx + 2] = 0;   // Blue
                    }
                } else {  // Horizontal seam
                    for (int x = 0; x < img.width; x++) {
                        int seam_y = seams[i][x];
                        int idx = (seam_y * img.width + x) * 3;
                        overlay_image[idx] = 255;     // Red
                        overlay_image[idx + 1] = 0;   // Green
                        overlay_image[idx + 2] = 0;   // Blue
                    }
                }
                
                char overlay_filename[256];
                sprintf(overlay_filename, "data/seam_path_step_%d.png", i + 1);
                Image overlay_img = {img.width, img.height, overlay_image};
                save_image(overlay_filename, overlay_img, 3);
                printf("Seam path overlay saved: %s\n", overlay_filename);
                free(overlay_image);
            }
            
            free(seams[i]);
        }
        
        free(seams);
        free(seam_lengths);
        
    } else { // seam removal logic
        for (int i = 0; i < num_seams; i++) {
            compute_energy(&img, device_energy);
            printf("Energy map computed on GPU for seam %d.\n", i + 1);
            
            int *seam = remove_seam_with_path(&img, device_energy, direction);
            printf("%s seam %d removed.\n", direction == 0 ? "Vertical" : "Horizontal", i + 1);
            
            // Save the seam path overlay if enabled
            if (save_seam_path) {
                // Store original dimensions before any seam removals
                static int original_width = -1;
                static int original_height = -1;
                if (original_width == -1) {
                    original_width = img.width;
                    original_height = img.height;
                }
                unsigned char *overlay_image = (unsigned char *)malloc(original_width * original_height * 3);
                memset(overlay_image, 0, original_width * original_height * 3);  // Initialize to black
                memcpy(overlay_image, img.data, img.width * img.height * 3);

                if (direction == 0) {  // Vertical seam
                    for (int y = 0; y < img.height; y++) {
                        int seam_x = seam[y];
                        int idx = (y * img.width + seam_x) * 3;
                        overlay_image[idx] = 255;     // Red
                        overlay_image[idx + 1] = 0;   // Green
                        overlay_image[idx + 2] = 0;   // Blue
                    }
                } else {  // Horizontal seam
                    for (int x = 0; x < img.width; x++) {
                        int seam_y = seam[x];
                        int idx = (seam_y * img.width + x) * 3;
                        overlay_image[idx] = 255;     // Red
                        overlay_image[idx + 1] = 0;   // Green
                        overlay_image[idx + 2] = 0;   // Blue
                    }
                }

                char overlay_filename[256];
                sprintf(overlay_filename, "data/seam_path_step_%d.png", i + 1);
                Image overlay_img = {img.width, img.height, overlay_image};
                save_image(overlay_filename, overlay_img, 3);
                printf("Seam path overlay saved: %s\n", overlay_filename);
                free(overlay_image);
            }
            
            free(seam);
        }
    }
    
    // Save the resulting image
    save_image(output_image_path, img, 3);
    printf("Output image saved: %s\n", output_image_path);
    
    // Cleanup
    free_image(img);
    free_device_memory(device_energy);
    
    return 0;
}