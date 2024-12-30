#ifndef UTILS_H
#define UTILS_H

// Image structure
typedef struct {
    int width;
    int height;
    unsigned char *data;
} Image;

// Function prototypes
Image load_image(const char *filename);
void save_image(const char *filename, Image img, int channels);  
void free_image(Image img);

#endif // UTILS_H
