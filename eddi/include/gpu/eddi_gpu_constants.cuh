/**
 * @file eddi_gpu_constants.cuh
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2025-05-19
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#ifndef __EDDI_GPU_CONSTANTS_CUH__
#define __EDDI_GPU_CONSTANTS_CUH__

__constant__ float c_origin_x;
__constant__ float c_origin_y;
__constant__ float c_origin_z;

__constant__ size_t c_x_size;
__constant__ size_t c_y_size;
__constant__ size_t c_z_size;

__constant__ float c_dx;
__constant__ float c_dy;
__constant__ float c_dz;

__constant__ size_t c_nc_x;
__constant__ size_t c_nc_y;
__constant__ size_t c_nc_z;

__constant__ float c_cx;
__constant__ float c_cy;
__constant__ float c_cz;

#endif // __EDDI_GPU_CONSTANTS_CUH__