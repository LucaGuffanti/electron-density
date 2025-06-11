/**
 * @file eddi_gpu_error_check.h
 * @author Luca Guffanti 
 * @brief Contains code used to check functions calls on the GPU
 */

#ifndef __EDDI_GPU_ERROR_CHECK_H__
#define __EDDI_GPU_ERROR_CHECK_H__

#include "cuda.h"
#include "cuda_runtime.h"
#include "eddi_base_includes.h"

EDDI_HEADER_OPEN

void check_error(cudaError_t error, char* msg);

EDDI_HEADER_CLOSE

#endif // __EDDI_GPU_ERROR_CHECK_H__

