/**
 * @file eddi_gpu_density.cuh
 * @author Luca Guffanti
 * @brief Density functions to be used on the device
 */

#ifndef __EDDI_GPU_DENSITY_CUH__
#define __EDDI_GPU_DENSITY_CUH__

#define EDDI_GPU_N_SLATER_DENSITIES 26
typedef float(*slater_density)(float r, float theta, float psi);

extern __device__ slater_density slater_densities[EDDI_GPU_N_SLATER_DENSITIES];

#endif // __EDDI_GPU_DENSITY_CUH__