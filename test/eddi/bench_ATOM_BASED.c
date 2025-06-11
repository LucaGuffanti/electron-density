#include "eddi_gpu_molecule.h"
#include "eddi_gpu_density_field.h"
#include "eddi_molecule_readers.h"
#include "eddi_density_writers.h"
#include "cuda.h"

int main(int argc, char** argv)
{
    cudaDeviceReset();
    eddi_molecule_boundary_t boundaries;
    eddi_gpu_molecule_t device_molecule;
    
    eddi_gpu_read_pdb(argv[1], &device_molecule, &boundaries);
    // eddi_gpu_print_molecule(&device_molecule);

    eddi_gpu_density_field_t device_density_field;


    eddi_real_t res = 1.0;
    eddi_cl_info_t info = {.cx = 12.0, .cy = 12.0, .cz = 12.0};
    eddi_gpu_init_field_from_molecule_binned(&device_density_field, &boundaries, 30.0, res, res, res, &info);
    eddi_gpu_compute_density_field_ATOMS(&device_density_field, &device_molecule);

    eddi_gpu_free_device_molecule(&device_molecule);
    eddi_free_atomic_data();
    eddi_gpu_free_density_field(&device_density_field);


    return 0;
}