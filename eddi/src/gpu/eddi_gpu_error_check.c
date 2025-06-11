#include "eddi_gpu_error_check.h"

void check_error(cudaError_t error, char* msg)
{
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n. %s", cudaGetErrorString(error), msg);
        exit(EXIT_FAILURE);
    }
}
