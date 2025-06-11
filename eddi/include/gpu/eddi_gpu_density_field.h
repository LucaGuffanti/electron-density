/**
 * @file eddi_gpu_density_field.h
 * @author Luca Guffanti
 * @brief Defintions of functions for management of density field
 */

#ifndef __EDDI_GPU_DENSITY_FIELD_H__
#define __EDDI_GPU_DENSITY_FIELD_H__

#include "eddi_gpu_types.h"
#include "eddi_macro.h"
#include "eddi_density_field.h"
#include "eddi_gpu_error_check.h"
#include "string.h"
#include "cuda.h"

EDDI_HEADER_OPEN

bool eddi_gpu_new_density_field(
    eddi_gpu_density_field_t* density_field,
    const eddi_real_t dx,
    const eddi_real_t dy,
    const eddi_real_t dz,
    const eddi_point_t* origin,
    const eddi_size_t n_x,
    const eddi_size_t n_y,
    const eddi_size_t n_z
);

bool eddi_gpu_init_field_from_molecule(
    eddi_gpu_density_field_t* density_field,
    eddi_molecule_boundary_t* boundaries,
    const eddi_real_t padding,
    const eddi_real_t dx,
    const eddi_real_t dy,
    const eddi_real_t dz
);

bool eddi_gpu_init_field_from_molecule_with_binning(
    eddi_gpu_density_field_t* density_field,
    eddi_molecule_boundary_t* boundaries,
    const eddi_real_t padding,
    const eddi_cl_info_t* bin_info,
    const eddi_size_t total_points_per_bin
);

bool eddi_gpu_init_field_from_molecule_binned(
    eddi_gpu_density_field_t* density,
    eddi_molecule_boundary_t* boundaries,
    const eddi_real_t padding,
    const eddi_real_t dx,
    const eddi_real_t dy,
    const eddi_real_t dz,
    const eddi_cl_info_t* bin_info
);

void _load_origin_in_constant_memory(const eddi_point_t* origin);

void _load_field_data_in_constant_memory(
    const eddi_size_t x_size,
    const eddi_size_t y_size,
    const eddi_size_t z_size,
    const eddi_real_t dx,
    const eddi_real_t dy,
    const eddi_real_t dz
);

void _load_cell_list_data_in_constant_memory(
    const eddi_size_t nc_x,
    const eddi_size_t nc_y,
    const eddi_size_t nc_z,
    const eddi_real_t cx,
    const eddi_real_t cy,
    const eddi_real_t cz
);

void eddi_gpu_print_origin();

void  eddi_gpu_compute_density_field_POINTS(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule);

void eddi_gpu_compute_density_field_ATOMS(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule);

void eddi_gpu_get_density_field(const eddi_gpu_density_field_t* device_field, eddi_density_field_t* host_field);

void eddi_gpu_compute_density_field_CELL_LIST(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info);

void ONE_THREAD_PER_POINT_eddi_gpu_compute_density_field_cell_list_previous_cutoff(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info);

void ONE_BLOCK_PER_BIN_eddi_gpu_compute_density_field_cell_list_previous_cutoff(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info);

void OPT_ONE_BLOCK_PER_BIN_eddi_gpu_compute_density_field_cell_list_previous_cutoff(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info);



void ONE_BLOCK_PER_POINT_eddi_gpu_compute_density_field_cell_list_previous_cutoff(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info);



void eddi_gpu_free_density_field(eddi_gpu_density_field_t* density_field);

EDDI_HEADER_CLOSE

#endif // __EDDI_GPU_DENSITY_FIELD_H__