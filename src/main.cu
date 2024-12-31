#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "gpu_memory.h"
#include "energy.h"
#include "seam_carving.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s -i <input_image> -o <output_image> [-n <number_of_seams>] [--debug] [--save-seams] [--horizontal]\n", argv[0]);
        return 1;
    }

    // Default values for options
    int num_seams = 1; 
    int save_seam_path = 0; 
    int direction = 0; // 0 for vertical, 1 for horizontal
    char *input_image_path = NULL;
    char *output_image_path = NULL;

    // Parse CLI arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_image_path = argv[++i]; // Set input image path
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_image_path = argv[++i]; // Set output image path
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_seams = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--save-seams") == 0) {
            save_seam_path = 1;
        } else if (strcmp(argv[i], "--horizontal") == 0) {
            direction = 1; // Set direction to horizontal
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printf("Usage: %s -i <input_image> -o <output_image> [-n <number_of_seams>] [--debug] [--save-seams] [--horizontal]\n", argv[0]);
            return 1;
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

    // Seam removal loop
    for (int i = 0; i < num_seams; i++) {
        // Compute the energy map
        compute_energy(&img, device_energy);
        printf("Energy map computed on GPU for seam %d.\n", i + 1);

        // Identify and remove one seam based on direction
        int *seam = remove_seam_with_path(&img, device_energy, direction);
        printf("%s seam %d removed.\n", direction == 0 ? "Vertical" : "Horizontal", i + 1);

        // Save the seam path overlay if enabled
        if (save_seam_path) {
            unsigned char *overlay_image = (unsigned char *)malloc(img.width * img.height * 3);
            memcpy(overlay_image, img.data, img.width * img.height * 3);

            for (int y = 0; y < img.height; y++) {
                int seam_x = seam[y];
                int idx = (y * img.width + seam_x) * 3;
                overlay_image[idx] = 255;     // Red channel
                overlay_image[idx + 1] = 0;   // Green
                overlay_image[idx + 2] = 0;   // Blue
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

    // Save the resulting image
    save_image(output_image_path, img, 3);
    printf("Output image saved: %s\n", output_image_path);

    // Free resources
    free_image(img);
    free_device_memory(device_energy);

    return 0;
}
