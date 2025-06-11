/**
 * @file eddi_gpu_molecule_kernel.cuh
 * @author Luca Guffanti
 * @brief Contains kernels used by the GPU to work on the molecule
 */

#ifndef __EDDI_GPU_MOLECULE_KERNEL_CUH__
#define __EDDI_GPU_MOLECULE_KERNEL_CUH__

#include "cuda_runtime.h"
#include "eddi_gpu_types.h"

__global__ void _gpu_print_molecule(float4* atoms, size_t n_atoms);


#endif //__EDDI_GPU_MOLECULE_KERNEL_CUH__