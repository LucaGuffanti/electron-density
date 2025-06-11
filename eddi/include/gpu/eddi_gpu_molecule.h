/**
 * @file eddi_gpu_molecule.h
 * @author Luca Guffanti
 * @brief Contains the molecule description for a molecule on GPU
 * @version 0.1
 * @date 2025-05-15
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef __EDDI_GPU_MOLECULE_H__
#define __EDDI_GPU_MOLECULE_H__

#include "eddi_macro.h"
#include "eddi_gpu_error_check.h"
#include "eddi_gpu_types.h"
#include "eddi_atomic_data.h"
#include "eddi_molecule.h"
#include "cuda.h"
#include "stdio.h"

EDDI_HEADER_OPEN

bool eddi_molecule_cpu_to_gpu(const eddi_molecule_t* cpu_molecule, eddi_gpu_molecule_t* gpu_molecule);

bool eddi_molecule_gpu_to_cpu(const eddi_gpu_molecule_t* gpu_molecule, eddi_molecule_t* cpu_molecule);

// ===== FUNCTIONS THAT TARGET THE GPU =====

bool eddi_gpu_read_pdb(const char* filename, eddi_gpu_molecule_t* device_molecule, eddi_molecule_boundary_t* boundaries);

bool eddi_gpu_read_coord(const char* filename, eddi_gpu_molecule_t* device_molecule, eddi_molecule_boundary_t* boundaries);

eddi_gpu_atom_t* eddi_gpu_offload_molecule(eddi_gpu_molecule_t* molecule);

void eddi_gpu_free_device_molecule(eddi_gpu_molecule_t* device_molecule);

void eddi_gpu_free_host_molecule(eddi_gpu_molecule_t* host_molecule);

void eddi_gpu_print_molecule(eddi_gpu_molecule_t* molecule);

eddi_gpu_molecule_t* eddi_gpu_unload_molecule(eddi_gpu_molecule_t* device_molecule);

EDDI_HEADER_CLOSE

#endif // __EDDI_GPU_MOLECULE_H__