#include "eddi_gpu_molecule.h"
#include "eddi_gpu_density_field.h"
#include "eddi_molecule_readers.h"
#include "eddi_density_writers.h"


int main(int argc, char** argv)
{
    eddi_molecule_boundary_t boundaries;
    eddi_gpu_molecule_t device_molecule;

    eddi_gpu_read_coord(argv[1], &device_molecule, &boundaries);

    eddi_gpu_density_field_t device_density_field;

    eddi_real_t res = 5.0;
    eddi_real_t c_res = 20.0;
    eddi_cl_info_t info = {.cx = c_res, .cy = c_res, .cz = c_res};
    eddi_gpu_init_field_from_molecule_binned(&device_density_field, &boundaries, 20.0, res, res, res, &info);
    // printf("Computed\n");
    OPT_ONE_BLOCK_PER_BIN_eddi_gpu_compute_density_field_cell_list_previous_cutoff(&device_density_field, &device_molecule, &info);

    eddi_molecule_t out_mol;
    eddi_density_field_t out_field;

    eddi_gpu_get_density_field(&device_density_field, &out_field);
    printf("Field\n");
    eddi_molecule_gpu_to_cpu(eddi_gpu_unload_molecule(&device_molecule), &out_mol);
    printf("Molecule\n");

    eddi_write_gaussian_cube("cell_list.cube", &out_field, &out_mol);
    eddi_write_binary("cell_list.bin", &out_field);
    // =====================================================


    eddi_gpu_free_device_molecule(&device_molecule);
    eddi_free_atomic_data();


    // =============================================================



    return EDDI_RETURN_SUCCESS;
}