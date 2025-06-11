/**
 * @file eddi_gpu_types.h
 * @author Luca Guffanti
 * @brief Define the types for molecular data to be offloaded on the gpu
 */

#ifndef __EDDI_GPU_TYPES_H__
#define __EDDI_GPU_TYPES_H__

#include "eddi_base_includes.h"
#include "eddi_density_field.h"
#include "eddi_macro.h"

EDDI_HEADER_OPEN

/**
 * @brief Atomic information on GPU
 * .x = x position
 * .y = y position
 * .z = z position
 * .w = atomic number
 */
typedef float4 eddi_gpu_atom_t;

/**
 * @brief Molecule information on GPU
 */
typedef struct eddi_gpu_molecule
{
    eddi_gpu_atom_t* atoms;
    eddi_size_t n_atoms;

} eddi_gpu_molecule_t;

typedef struct eddi_gpu_atom_list
{
    eddi_gpu_atom_t atom;
    struct eddi_gpu_atom_list* next;

} eddi_gpu_atom_list_t;

typedef eddi_density_field_t eddi_gpu_density_field_t;


EDDI_HEADER_CLOSE

#endif // __EDDI_GPU_TYPES_H__