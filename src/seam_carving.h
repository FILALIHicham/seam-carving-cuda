#ifndef SEAM_CARVING_H
#define SEAM_CARVING_H

#include "utils.h" 

// Removes one seam from the image
int* remove_seam_with_path(Image *img, float *device_energy, int direction, bool optimized);

int** find_k_seams(Image *img, float *device_energy, int k, int direction, int *seam_lengths, bool optimized);

// Adds one seam to the image
void insert_seam(Image *img, int *seam, int direction);

#endif // SEAM_CARVING_H
