#include "utils.h"
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"          // Library for loading images
#include "stb_image_write.h"    // Library for saving images
#include <stdio.h>
#include <stdlib.h>

// Function to load an image from a file
Image load_image(const char *filename) {
    Image img = {0}; // Initialize an empty Image struct
    int channels;    // Number of color channels (RGB = 3, RGBA = 4)

    // Load the image using stb_image
    img.data = stbi_load(filename, &img.width, &img.height, &channels, 3); 
    if (!img.data) {
        fprintf(stderr, "Error: Unable to load image %s\n", filename);
        exit(EXIT_FAILURE); // Exit if loading fails
    }

    printf("Loaded image: %s (%dx%d, %d channels)\n", filename, img.width, img.height, channels);
    return img; 
}


// Function to save an image to a file
void save_image(const char *filename, Image img, int channels) {
    if (!img.data) {
        fprintf(stderr, "Error: No image data to save\n");
        return;
    }

    // Save the image as a PNG file
    if (!stbi_write_png(filename, img.width, img.height, channels, img.data, img.width * channels)) {
        fprintf(stderr, "Error: Unable to save image %s\n", filename);
        exit(EXIT_FAILURE);
    }

    printf("Saved image: %s (%dx%d, %d channels)\n", filename, img.width, img.height, channels);
}

// Function to free memory allocated for an image
void free_image(Image img) {
    if (img.data) {
        free(img.data); 
        printf("Freed image data\n");
    }
}

// Save seam overlay on the image
void save_seam_overlay(const Image *img, const int *seam, int direction, const char *filename) {
    // Allocate memory for the overlay image
    unsigned char *overlay_data = (unsigned char *)malloc(img->width * img->height * 3);
    if (!overlay_data) {
        fprintf(stderr, "Error: Failed to allocate memory for seam overlay.\n");
        exit(1);
    }

    // Copy the original image data
    memcpy(overlay_data, img->data, img->width * img->height * 3);

    // Draw the seam in red
    if (direction == 0) {  // Vertical seam
        for (int y = 0; y < img->height; y++) {
            int seam_x = seam[y];
            int idx = (y * img->width + seam_x) * 3;
            overlay_data[idx] = 255;      // Red
            overlay_data[idx + 1] = 0;    // Green
            overlay_data[idx + 2] = 0;    // Blue
        }
    } else {  // Horizontal seam
        for (int x = 0; x < img->width; x++) {
            int seam_y = seam[x];
            int idx = (seam_y * img->width + x) * 3;
            overlay_data[idx] = 255;      // Red
            overlay_data[idx + 1] = 0;    // Green
            overlay_data[idx + 2] = 0;    // Blue
        }
    }

    // Save the overlay image
    Image overlay_img = {img->width, img->height, overlay_data};
    save_image(filename, overlay_img, 3);

    // Free the overlay data
    free(overlay_data);

    printf("Seam overlay saved to %s\n", filename);
}