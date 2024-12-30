#ifndef SEAM_CARVING_H
#define SEAM_CARVING_H

#include "utils.h" 

// Removes one vertical seam from the image
int* remove_seam_with_path(Image *img, float *device_energy);

#endif // SEAM_CARVING_H
