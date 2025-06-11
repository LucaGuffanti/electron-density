#include "eddi_gpu_molecule.h"
#include "eddi_gpu_density_field.h"
#include "eddi_molecule_readers.h"
#include "eddi_density_writers.h"


int main(int argc, char** argv)
{
    // eddi_molecule_t molecule_cpu_format;
    // eddi_gpu_molecule_t molecule_gpu_format;
    eddi_molecule_boundary_t boundaries;

    // eddi_read_pdb(argv[1], &molecule_cpu_format);
    // eddi_molecule_cpu_to_gpu(&molecule_cpu_format, &molecule_gpu_format);
    // printf("Changed data format\n");

    // eddi_gpu_atom_t* d_molecule = eddi_gpu_offload_molecule(&molecule_gpu_format);
    // eddi_gpu_molecule_t mol = {.atoms = d_molecule, .n_atoms = molecule_gpu_format.n_atoms};
    // printf("Offloaded\n");

    // eddi_free_molecule(&molecule_cpu_format);
    // eddi_gpu_free_device_molecule(&mol);
    // eddi_gpu_free_host_molecule(&molecule_gpu_format);
    // printf("Freed\n");
    

    // ============================================================= 

    eddi_gpu_molecule_t device_molecule;
    
    eddi_gpu_read_pdb(argv[1], &device_molecule, &boundaries);
    // eddi_gpu_print_molecule(&device_molecule);

    eddi_gpu_density_field_t device_density_field;

    // eddi_gpu_print_origin();

    eddi_real_t res = 1.0;
    eddi_cl_info_t info = {.cx = 12.0, .cy = 12.0, .cz = 12.0};
    eddi_gpu_init_field_from_molecule_binned(&device_density_field, &boundaries, 40.0, res, res, res, &info);
    // printf("Computed\n");
    ONE_BLOCK_PER_BIN_eddi_gpu_compute_density_field_cell_list_previous_cutoff(&device_density_field, &device_molecule, &info);

    eddi_molecule_t out_mol;
    eddi_density_field_t out_field;
    
    eddi_gpu_get_density_field(&device_density_field, &out_field);
    printf("Field\n");
    // eddi_gpu_print_molecule(&device_molecule);
    eddi_molecule_gpu_to_cpu(eddi_gpu_unload_molecule(&device_molecule), &out_mol);
    printf("Molecule\n");
    
    eddi_write_gaussian_cube("cell_list1.cube", &out_field, &out_mol);
    eddi_write_binary("cell_list.bin", &out_field);
    // =====================================================
    

    eddi_gpu_free_device_molecule(&device_molecule);
    eddi_free_atomic_data();

    
    // =============================================================



    return EDDI_RETURN_SUCCESS;
}