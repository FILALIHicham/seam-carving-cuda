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