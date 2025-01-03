#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"
#include "gpu_memory.h"
#include "energy.h"
#include "seam_carving.h"

int main(int argc, char *argv[]) {
    if (argc < 4) {
        printf("Usage: %s -i <input_image> -o <output_image> [-n <number_of_seams>] [--insert] [--save-seams] [--horizontal] [--target widthxheight]\n", argv[0]);
        return 1;
    }

    // Default values for options
    int num_seams = 1;
    int save_seam_path = 0;
    int direction = 0; // 0 for vertical, 1 for horizontal
    int insert_mode = 0; // 0 for removal, 1 for insertion
    int target_width = -1, target_height = -1;
    char *input_image_path = NULL;
    char *output_image_path = NULL;

    // Flags to check for invalid parameter combinations
    int target_mode = 0;
    int seam_mode = 0;

    // Parse CLI arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) {
            input_image_path = argv[++i];
        } else if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_image_path = argv[++i];
        } else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_seams = atoi(argv[++i]);
            seam_mode = 1;
        } else if (strcmp(argv[i], "--save-seams") == 0) {
            save_seam_path = 1;
        } else if (strcmp(argv[i], "--horizontal") == 0) {
            direction = 1;
            seam_mode = 1;
        } else if (strcmp(argv[i], "--insert") == 0) {
            insert_mode = 1;
            seam_mode = 1;
        } else if (strcmp(argv[i], "--target") == 0 && i + 1 < argc) {
            char *target_str = argv[++i];
            if (sscanf(target_str, "%dx%d", &target_width, &target_height) != 2) {
                printf("Error: Invalid target format. Use --target widthxheight (example: 800x600).\n");
                return 1;
            }
            target_mode = 1;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printf("Usage: %s -i <input_image> -o <output_image> [-n <number_of_seams>] [--insert] [--save-seams] [--horizontal] [--target widthxheight]\n", argv[0]);
            return 1;
        }
    }

    // Ensure mutually exclusive modes
    if (target_mode && seam_mode) {
        printf("Error: --target cannot be combined with -n, --insert, or --horizontal.\n");
        return 1;
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

    // Handle target resizing
    if (target_mode) {
        int width_diff = target_width - img.width;
        int height_diff = target_height - img.height;

        printf("Target dimensions: %dx%d\n", target_width, target_height);
        printf("Width difference: %d, Height difference: %d\n", width_diff, height_diff);

        // Adjust width
        if (width_diff > 0) {
            printf("Inserting %d vertical seams...\n", width_diff);
            int **seams;
            int *seam_lengths = (int *)malloc(width_diff * sizeof(int));
            seams = find_k_seams(&img, device_energy, width_diff, 0, seam_lengths);

            for (int i = 0; i < width_diff; i++) {
                insert_seam(&img, seams[i], 0);
                printf("Vertical seam %d inserted.\n", i + 1);

                if (save_seam_path) {
                    char overlay_filename[256];
                    sprintf(overlay_filename, "data/seam_path_vertical_step_%d.png", i + 1);
                    save_seam_overlay(&img, seams[i], 0, overlay_filename);
                }

                free(seams[i]);
            }

            free(seams);
            free(seam_lengths);
        } else if (width_diff < 0) {
            printf("Removing %d vertical seams...\n", -width_diff);
            for (int i = 0; i < -width_diff; i++) {
                compute_energy(&img, device_energy);  
                int *seam = remove_seam_with_path(&img, device_energy, 0);
                printf("Vertical seam %d removed.\n", i + 1);

                if (save_seam_path) {
                    char overlay_filename[256];
                    sprintf(overlay_filename, "data/seam_path_vertical_step_%d.png", i + 1);
                    save_seam_overlay(&img, seam, 0, overlay_filename);
                }

                free(seam);
            }
        }

        free_device_memory(device_energy);
        allocate_device_memory((void **)&device_energy, img.width * img.height * sizeof(float));

        // Adjust height
        if (height_diff > 0) {
            printf("Inserting %d horizontal seams...\n", height_diff);
            int **seams;
            int *seam_lengths = (int *)malloc(height_diff * sizeof(int));
            seams = find_k_seams(&img, device_energy, height_diff, 1, seam_lengths);

            for (int i = 0; i < height_diff; i++) {
                insert_seam(&img, seams[i], 1);
                printf("Horizontal seam %d inserted.\n", i + 1);

                if (save_seam_path) {
                    char overlay_filename[256];
                    sprintf(overlay_filename, "data/seam_path_horizontal_step_%d.png", i + 1);
                    save_seam_overlay(&img, seams[i], 1, overlay_filename);
                }

                free(seams[i]);
            }

            free(seams);
            free(seam_lengths);
        } else if (height_diff < 0) {
            printf("Removing %d horizontal seams...\n", -height_diff);
            for (int i = 0; i < -height_diff; i++) {
                compute_energy(&img, device_energy);
                int *seam = remove_seam_with_path(&img, device_energy, 1);
                printf("Horizontal seam %d removed.\n", i + 1);

                if (save_seam_path) {
                    char overlay_filename[256];
                    sprintf(overlay_filename, "data/seam_path_horizontal_step_%d.png", i + 1);
                    save_seam_overlay(&img, seam, 1, overlay_filename);
                }

                free(seam);
            }
        }
    } else if (insert_mode) {
        // Seam insertion logic
        printf("Starting seam insertion...\n");
        int **seams;
        int *seam_lengths = (int *)malloc(num_seams * sizeof(int));
        seams = find_k_seams(&img, device_energy, num_seams, direction, seam_lengths);

        for (int i = 0; i < num_seams; i++) {
            insert_seam(&img, seams[i], direction);
            printf("%s seam %d inserted.\n", direction == 0 ? "Vertical" : "Horizontal", i + 1);

            // Save seam path if enabled
            if (save_seam_path) {
                char overlay_filename[256];
                sprintf(overlay_filename, "data/seam_path_%s_step_%d.png", direction == 0 ? "vertical" : "horizontal", i + 1);
                save_seam_overlay(&img, seams[i], direction, overlay_filename);
            }

            free(seams[i]);
        }

        free(seams);
        free(seam_lengths);
    } else {
        // Seam removal logic
        printf("Starting seam removal...\n");
        for (int i = 0; i < num_seams; i++) {
            compute_energy(&img, device_energy);
            printf("Energy map computed on GPU for seam %d.\n", i + 1);

            int *seam = remove_seam_with_path(&img, device_energy, direction);
            printf("%s seam %d removed.\n", direction == 0 ? "Vertical" : "Horizontal", i + 1);

            // Save seam path if enabled
            if (save_seam_path) {
                char overlay_filename[256];
                sprintf(overlay_filename, "data/seam_path_%s_step_%d.png", direction == 0 ? "vertical" : "horizontal", i + 1);
                save_seam_overlay(&img, seam, direction, overlay_filename);
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
