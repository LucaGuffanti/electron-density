#include "eddi_gpu_molecule.h"
#include "eddi_gpu_molecule_kernel.cuh"

__global__ void _gpu_print_molecule(eddi_gpu_atom_t* atoms, eddi_size_t n_atoms)
{
    printf("Entered\n");
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(unsigned int i = tid; i < n_atoms; i += blockDim.x * gridDim.x)
    {
        printf("Thread [%d]: atom %d - x: %f, y: %f, z: %f, w: %d\n", tid, i, atoms[i].x, atoms[i].y, atoms[i].z, (int) atoms[i].w);
    }
}

void eddi_gpu_print_molecule(eddi_gpu_molecule_t* molecule)
{

    printf("Printing molecule with %d atoms\n", molecule->n_atoms);
    _gpu_print_molecule<<<1, 256>>>(molecule->atoms, molecule->n_atoms);
    check_error(cudaDeviceSynchronize(), "cuda device synchronize - print molecule");
    cudaError_t err = cudaGetLastError();
    check_error(err, "cuda get last error - print molecule");
}
