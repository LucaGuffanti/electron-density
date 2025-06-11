/**
 * @file eddi_gpu_density_field_kernel.cuh
 * @author Luca Guffanti
 * @brief Functions to be executed on the device
 */

#ifndef __EDDI_GPU_DENSITY_FIELD_KERNEL_CUH__
#define __EDDI_GPU_DENSITY_FIELD_KERNEL_CUH__

#include "eddi_gpu_constants.cuh"
#include "eddi_gpu_density_functions.cuh"
#include "cuda_runtime.h"


__global__ void _gpu_print_origin()
{
    unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    printf("Thread %d %f %f %f\n", tid, c_origin_x, c_origin_y, c_origin_z);
}

__device__ float slater_functions(int z, float r, float theta, float phi)
{
    switch (z)
    {
    case 1:
    {
        // Slater wavefunction for atomic number 1
        return 0.3183098861837909 * exp(-2 * 1.0 * r);
    }

    case 2:
    {
        // Slater wavefunction for atomic number 2
        return 2 * 1.56385647082292 * exp(-2 * 1.7 * r);
    }
    case 3:
    {
        // Slater wavefunction for atomic number 3
        return 2 * 6.26529348975552 * exp(-2 * 2.7 * r) + 0.01231106589279565 * pow(r, 2.0) * exp(-2 * 0.65 * r);
    }

    case 4:
    {
        // Slater wavefunction for atomic number 4
        return 2 * 16.12335066486755 * exp(-2 * 3.7 * r) + 2 * 0.09348715662341694 * pow(r, 2.0) * exp(-2 * 0.9750000000000001 * r);
    }

    case 5:
    {
        // Slater wavefunction for atomic number 5
        return 2 * 33.04788731336516 * exp(-2 * 4.7 * r) + 3 * 0.39395410856946067 * pow(r, 2.0) * exp(-2 * 1.3 * r);
    }

    case 6:
    {
        // Slater wavefunction for atomic number 6
        return 2 * 58.94876275201326 * exp(-2 * 5.7 * r) + 4 * 1.2022525285933259 * pow(r, 2.0) * exp(-2 * 1.625 * r);
    }

    case 7:
    {
        // Slater wavefunction for atomic number 7
        return 2 * 95.73583629830304 * exp(-2 * 6.7 * r) + 5 * 2.9915890119493427 * pow(r, 2.0) * exp(-2 * 1.9500000000000002 * r);
    }

    case 8:
    {
        // Slater wavefunction for atomic number 8
        return 2 * 145.3189672691407 * exp(-2 * 7.7 * r) + 6 * 6.46600263938176 * pow(r, 2.0) * exp(-2 * 2.275 * r);
    }

    case 9:
    {
        // Slater wavefunction for atomic number 9
        return 2 * 209.6080149816861 * exp(-2 * 8.7 * r) + 7 * 12.606531474226806 * pow(r, 2.0) * exp(-2 * 2.6 * r);
    }

    case 10:
    {
        // Slater wavefunction for atomic number 10
        return 2 * 290.51283875301874 * exp(-2 * 9.7 * r) + 8 * 22.71737905948798 * pow(r, 2.0) * exp(-2 * 2.9250000000000003 * r);
    }

    case 11:
    {
        // Slater wavefunction for atomic number 11
        return 2 * 389.94329790024904 * exp(-2 * 10.7 * r) + 8 * 50.007109436657835 * pow(r, 2.0) * exp(-2 * 3.4250000000000003 * r) + 0.0016135320335943066 * pow(r, 4.0) * exp(-2 * 0.7333333333333331 * r);
    }

    case 12:
    {
        // Slater wavefunction for atomic number 12
        return 2 * 509.80925174048366 * exp(-2 * 11.7 * r) + 8 * 98.8387349088708 * pow(r, 2.0) * exp(-2 * 3.9250000000000003 * r) + 2 * 0.00987945178833327 * pow(r, 4.0) * exp(-2 * 0.9500000000000005 * r);
    }

    case 13:
    {
        // Slater wavefunction for atomic number 13
        return 2 * 652.0205595908079 * exp(-2 * 12.7 * r) + 8 * 180.00942628732454 * pow(r, 2.0) * exp(-2 * 4.425000000000001 * r) + 3 * 0.041619334990451416 * pow(r, 4.0) * exp(-2 * 1.1666666666666667 * r);
    }

    case 14:
    {
        // Slater wavefunction for atomic number 14
        return 2 * 818.487080768349 * exp(-2 * 13.7 * r) + 8 * 307.4397701413907 * pow(r, 2.0) * exp(-2 * 4.925000000000001 * r) + 4 * 0.13713727109030824 * pow(r, 4.0) * exp(-2 * 1.3833333333333335 * r);
    }

    case 15:
    {
        // Slater wavefunction for atomic number 15
        return 2 * 1011.1186745901936 * exp(-2 * 14.7 * r) + 8 * 498.57165616526265 * pow(r, 2.0) * exp(-2 * 5.425000000000001 * r) + 5 * 0.37975848643135146 * pow(r, 4.0) * exp(-2 * 1.6000000000000003 * r);
    }

    case 16:
    {
        // Slater wavefunction for atomic number 16
        return 2 * 1231.825200373448 * exp(-2 * 15.7 * r) + 8 * 774.7661644779737 * pow(r, 2.0) * exp(-2 * 5.925000000000001 * r) + 6 * 0.9238348519132616 * pow(r, 4.0) * exp(-2 * 1.8166666666666664 * r);
    }

    case 17:
    {
        // Slater wavefunction for atomic number 17
        return 2 * 1482.5165174352162 * exp(-2 * 16.7 * r) + 8 * 1161.7014531331229 * pow(r, 2.0) * exp(-2 * 6.425000000000001 * r) + 7 * 2.032954558631853 * pow(r, 4.0) * exp(-2 * 2.0333333333333337 * r);
    }

    case 18:
    {
        // Slater wavefunction for atomic number 18
        return 2 * 1765.1024850926005 * exp(-2 * 17.7 * r) + 8 * 1689.7706452760394 * pow(r, 2.0) * exp(-2 * 6.925000000000001 * r) + 8 * 4.129954204672846 * pow(r, 4.0) * exp(-2 * 2.25 * r);
    }

    case 19:
    {
        // Slater wavefunction for atomic number 19
        return 2 * 2081.4929626627027 * exp(-2 * 18.7 * r) + 8 * 2394.4797166230687 * pow(r, 2.0) * exp(-2 * 7.425000000000001 * r) + 8 * 10.862523867423388 * pow(r, 4.0) * exp(-2 * 2.5833333333333335 * r) + 2.9906737611881975e-05 * pow(r, 5.4) * exp(-2 * 0.5945945945945944 * r);
    }

    case 20:
    {
        // Slater wavefunction for atomic number 20
        return 2 * 2433.5978094626257 * exp(-2 * 19.7 * r) + 8 * 3316.845382815414 * pow(r, 2.0) * exp(-2 * 7.925000000000001 * r) + 8 * 25.402426141625856 * pow(r, 4.0) * exp(-2 * 2.9166666666666665 * r) + 2 * 0.00026309666655790993 * pow(r, 5.4) * exp(-2 * 0.7702702702702706 * r);
    }

    case 21:
    {
        // Slater wavefunction for atomic number 21
        return 2 * 2823.3268848094744 * exp(-2 * 20.7 * r) + 8 * 4503.792986750791 * pow(r, 2.0) * exp(-2 * 8.425 * r) + 8 * 54.18151395524859 * pow(r, 4.0) * exp(-2 * 3.25 * r) + 2 * 0.00040479679115536026 * pow(r, 5.4) * exp(-2 * 0.8108108108108107 * r) + 0.01414710605261292 * pow(r, 4.0) * exp(-2 * 1.0 * r);
    }

    case 22:
    {
        // Slater wavefunction for atomic number 22
        return 2 * 3252.590048020349 * exp(-2 * 21.7 * r) + 8 * 6008.554385940561 * pow(r, 2.0) * exp(-2 * 8.925 * r) + 8 * 107.31936045217444 * pow(r, 4.0) * exp(-2 * 3.5833333333333335 * r) + 2 * 0.0006098558208557976 * pow(r, 5.4) * exp(-2 * 0.8513513513513509 * r) + 2 * 0.05583016062783766 * pow(r, 4.0) * exp(-2 * 1.2166666666666661 * r);
    }

    case 23:
    {
        // Slater wavefunction for atomic number 23
        return 2 * 3723.2971584123543 * exp(-2 * 22.7 * r) + 8 * 7891.06583987383 * pow(r, 2.0) * exp(-2 * 9.425 * r) + 8 * 200.02482190951136 * pow(r, 4.0) * exp(-2 * 3.9166666666666665 * r) + 2 * 0.0009014374909556084 * pow(r, 5.4) * exp(-2 * 0.891891891891892 * r) + 3 * 0.17583204016482312 * pow(r, 4.0) * exp(-2 * 1.4333333333333336 * r);
    }

    case 24:
    {
        // Slater wavefunction for atomic number 24
        return 2 * 4237.358075302591 * exp(-2 * 23.7 * r) + 8 * 10218.365897376594 * pow(r, 2.0) * exp(-2 * 9.925 * r) + 8 * 354.31547390255463 * pow(r, 4.0) * exp(-2 * 4.25 * r) + 2 * 0.0013094801520034804 * pow(r, 5.4) * exp(-2 * 0.9324324324324331 * r) + 4 * 0.47103721296316764 * pow(r, 4.0) * exp(-2 * 1.6499999999999997 * r);
    }

    case 25:
    {
        // Slater wavefunction for atomic number 25
        return 2 * 4796.682658008162 * exp(-2 * 24.7 * r) + 8 * 13064.993283967953 * pow(r, 2.0) * exp(-2 * 10.425 * r) + 8 * 601.0875231006424 * pow(r, 4.0) * exp(-2 * 4.583333333333333 * r) + 2 * 0.0018722314140677398 * pow(r, 5.4) * exp(-2 * 0.9729729729729734 * r) + 5 * 1.117210516657861 * pow(r, 4.0) * exp(-2 * 1.8666666666666671 * r);
    }

    case 26:
    {
        // Slater wavefunction for atomic number 26
        return 2 * 5403.180765846168 * exp(-2 * 25.7 * r) + 8 * 16513.384789216943 * pow(r, 2.0) * exp(-2 * 10.925 * r) + 8 * 982.5687970790284 * pow(r, 4.0) * exp(-2 * 4.916666666666667 * r) + 2 * 0.0026380387491179032 * pow(r, 5.4) * exp(-2 * 1.0135135135135134 * r) + 6 * 2.409788611299073 * pow(r, 4.0) * exp(-2 * 2.0833333333333335 * r);
    }

    default:
        return 0.0;
    }
}


__global__ void _gpu_density_field_if_else(
    float* density_field,
    float dx,
    float dy,
    float dz,
    size_t x_size,
    size_t y_size,
    size_t z_size,
    float4* atoms,
    size_t n_atoms
)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t points = c_x_size * c_y_size * c_z_size;
    const float max_distance = 144.0f;

    for (size_t i = tid; i < points; i += blockDim.x * gridDim.x) {
        const size_t it_z = i % c_z_size;
        const size_t it_y = (i / c_z_size) % c_y_size;
        const size_t it_x = (i / c_z_size) / c_y_size;

        const float cx = c_origin_x + it_x * c_dx;
        const float cy = c_origin_y + it_y * c_dy;
        const float cz = c_origin_z + it_z * c_dz;
        float local_density = 0.0f;
        for (size_t atom_idx = 0; atom_idx < n_atoms; ++atom_idx)
        {
            const float delta_x = cx - atoms[atom_idx].x;
            const float delta_y = cy - atoms[atom_idx].y;
            const float delta_z = cz - atoms[atom_idx].z;

            const float distance_squared = delta_x * delta_x
                                        + delta_y * delta_y
                                        + delta_z * delta_z;
            if (distance_squared < max_distance)
            {
                local_density += slater_functions((int) atoms[atom_idx].w, sqrt(distance_squared), 0.0f, 0.0f);
            }
        }
        density_field[i] = local_density;
    }
}

__global__ void _gpu_density_field_func_ptr(
    float* density_field,
    float dx,
    float dy,
    float dz,
    size_t x_size,
    size_t y_size,
    size_t z_size,
    float4* atoms,
    size_t n_atoms
)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t points = x_size * y_size * z_size;
    const float max_distance = 144.0f;

    for (size_t i = tid; i < points; i += blockDim.x * gridDim.x) {
        const size_t it_z = i % z_size;
        const size_t it_y = (i / z_size) % y_size;
        const size_t it_x = (i / z_size) / y_size;

        const float cx = c_origin_x + it_x * dx;
        const float cy = c_origin_y + it_y * dy;
        const float cz = c_origin_z + it_z * dz;
        float local_density = 0.0f;
        for (size_t atom_idx = 0; atom_idx < n_atoms; ++atom_idx)
        {
            const float delta_x = cx - atoms[atom_idx].x;
            const float delta_y = cy - atoms[atom_idx].y;
            const float delta_z = cz - atoms[atom_idx].z;

            const float distance_squared = delta_x * delta_x
                                        + delta_y * delta_y
                                        + delta_z * delta_z;
            if (distance_squared < max_distance)
            {
                local_density += slater_densities[(int)(atoms[atom_idx].w)-1](sqrt(distance_squared), 0.0f, 0.0f);
            }
        }
        density_field[i] = local_density;
    }
}


__global__ void _gpu_density_partition_atoms(
    float4* partitioned_molecule,
    size_t* occupations,
    const size_t max_occupation,
    const float4* atoms,
    const size_t n_atoms
)
{

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t atoms_in_shared = 512;
    const size_t n_cells = c_nc_x * c_nc_y * c_nc_z;
    __shared__ float4 local_atoms[atoms_in_shared];
    size_t batches = (n_atoms + atoms_in_shared - 1) / atoms_in_shared;

    // if (tid == 0)
    //     printf("Batches: %d\n", batches);

    const size_t current_cell = tid;

    if (current_cell >= n_cells) return;

    size_t cell_z = current_cell % c_nc_z;
    size_t cell_y = (current_cell / c_nc_z) % c_nc_y;
    size_t cell_x = (current_cell / (c_nc_y * c_nc_z));

    // printf("Cell coordinates: z = %lu, y = %lu, x = %lu\n", cell_z, cell_y, cell_x);

    const float x_min = c_origin_x + cell_x * c_cx;
    const float x_max = x_min + c_cx;
    const float y_min = c_origin_y + cell_y * c_cy;
    const float y_max = y_min + c_cy;
    const float z_min = c_origin_z + cell_z * c_cz;
    const float z_max = z_min + c_cz;

    size_t local_occupation = 0;

    for (size_t current_batch = 0; current_batch < batches; ++current_batch)
    {
        __syncthreads();

        for (size_t local_tid = threadIdx.x; local_tid < atoms_in_shared; local_tid += blockDim.x)
        {
            size_t loaded_atom_id = local_tid + current_batch * atoms_in_shared;
            if (loaded_atom_id < n_atoms)
                local_atoms[local_tid] = atoms[loaded_atom_id];
        }

        __syncthreads();

        for (size_t atom_id = 0; atom_id < atoms_in_shared && current_batch * atoms_in_shared + atom_id < n_atoms; ++atom_id)
        {
            float4 atom = local_atoms[atom_id];
            // if (tid == 0) {
            //     printf("Considering atom at index %lu with coordinates (%f, %f, %f)\n",
            //            current_batch * atoms_in_shared + atom_id,
            //            atom.x, atom.y, atom.z);
            // }
            if (atom.x >= x_min && atom.x < x_max &&
                atom.y >= y_min && atom.y < y_max &&
                atom.z >= z_min && atom.z < z_max

            )
            {

                if (local_occupation < max_occupation) {
                    // printf("Thread %d adds atom %lu to its cell -> %d (Cell coordinates: [%f, %f, %f], Intervals: x[%f, %f], y[%f, %f], z[%f, %f])\n",
                    //        tid,
                    //        current_batch * atoms_in_shared + atom_id,
                    //        local_occupation,
                    //        x_min, y_min, z_min,
                    //        x_min, x_max,
                    //        y_min, y_max,
                    //        z_min, z_max);
                    partitioned_molecule[max_occupation * current_cell + local_occupation] = local_atoms[atom_id];
                    local_occupation += 1;
                } else
                {
                    printf("Error: Exceeded maximum occupation for cell at thread %d\n", tid);
                    return;
                }
            }
        }
    }
    occupations[current_cell] = local_occupation;
    return;
}

__global__ void _gpu_density_partition_atoms_ATOM_BASED(
    float4* partitioned_molecule,
    size_t* occupation,
    const size_t max_occupation,
    const float4* atoms,
    const size_t n_atoms
)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n_atoms) return;

    float4 atom = atoms[tid];

    size_t bin_x = (atom.x - c_origin_x) / c_cx;
    size_t bin_y = (atom.y - c_origin_y) / c_cy;
    size_t bin_z = (atom.z - c_origin_z) / c_cz;

    size_t bin_id = bin_x * (c_nc_y * c_nc_z) + bin_y * c_nc_z + bin_z;

    
    unsigned int prev_occ = atomicAdd((unsigned int*)&occupation[bin_id], 1);
    if (prev_occ < max_occupation)
    {
        partitioned_molecule[max_occupation * bin_id + prev_occ] = atom;
    }
    else
    {
        printf("Error: Exceeded maximum occupation for bin %lu by atom %lu\n", bin_id, tid);
    }

}


__global__ void _gpu_density_partition_atoms_3D_GRID(
    float4* partitioned_molecule,
    size_t* occupations,
    const size_t max_occupation,
    const float4* atoms,
    const size_t n_atoms
)
{

    size_t tid_x = threadIdx.x + blockIdx.x * blockDim.x;
    size_t tid_y = threadIdx.y + blockIdx.y * blockDim.y;
    size_t tid_z = threadIdx.z + blockIdx.z * blockDim.z;

    size_t tid = tid_z + tid_y * gridDim.z * blockDim.z + tid_x * gridDim.z * blockDim.z * gridDim.y * blockDim.y;

    const size_t atoms_in_shared = 512;
    const size_t n_cells = c_nc_x * c_nc_y * c_nc_z;
    __shared__ float4 local_atoms[atoms_in_shared];
    size_t batches = (n_atoms + atoms_in_shared - 1) / atoms_in_shared;

    // if (tid == 0)
    //     printf("Batches: %d\n", batches);

    const size_t current_cell = tid;

    if (current_cell >= n_cells) return;

    // printf("Cell coordinates: z = %lu, y = %lu, x = %lu\n", cell_z, cell_y, cell_x);

    const float x_min = c_origin_x + tid_x * c_cx;
    const float x_max = x_min + c_cx;
    const float y_min = c_origin_y + tid_y * c_cy;
    const float y_max = y_min + c_cy;
    const float z_min = c_origin_z + tid_z * c_cz;
    const float z_max = z_min + c_cz;

    size_t local_occupation = 0;

    for (size_t current_batch = 0; current_batch < batches; ++current_batch)
    {
        __syncthreads();

        for (size_t local_tid = tid; local_tid < atoms_in_shared && current_batch * atoms_in_shared + local_tid < n_atoms; local_tid += blockDim.x * blockDim.y * blockDim.z)
        {
            size_t loaded_atom_id = local_tid + current_batch * atoms_in_shared;
            if (loaded_atom_id < n_atoms)
                local_atoms[local_tid] = atoms[loaded_atom_id];
        }

        __syncthreads();

        for (size_t atom_id = 0; atom_id < atoms_in_shared && current_batch * atoms_in_shared + atom_id < n_atoms; ++atom_id)
        {
            float4 atom = local_atoms[atom_id];
            // if (tid == 0) {
            //     printf("Considering atom at index %lu with coordinates (%f, %f, %f)\n",
            //            current_batch * atoms_in_shared + atom_id,
            //            atom.x, atom.y, atom.z);
            // }
            if (atom.x >= x_min && atom.x < x_max &&
                atom.y >= y_min && atom.y < y_max &&
                atom.z >= z_min && atom.z < z_max

            )
            {

                if (local_occupation < max_occupation) {
                    // printf("Thread %d adds atom %lu to its cell -> %d (Cell coordinates: [%f, %f, %f], Intervals: x[%f, %f], y[%f, %f], z[%f, %f])\n",
                    //        tid,
                    //        current_batch * atoms_in_shared + atom_id,
                    //        local_occupation,
                    //        x_min, y_min, z_min,
                    //        x_min, x_max,
                    //        y_min, y_max,
                    //        z_min, z_max);
                    partitioned_molecule[max_occupation * current_cell + local_occupation] = local_atoms[atom_id];
                    local_occupation += 1;
                } else
                {
                    printf("Error: Exceeded maximum occupation for cell at thread %d\n", tid);
                    return;
                }
            }
        }
    }
    occupations[current_cell] = local_occupation;
    return;
}

__global__ void _gpu_density_field_complete_neighborhood(
    float* __restrict__ density_field,
    const float4* partitioned_molecule,
    size_t* occupations,
    const size_t max_occupation
)
{
    // The simplest approach is to use just the thread id to identify the point. From
    // that we access the complete neighborhood

    // This first versions serves as a way of undestanding how well the algorithm behaves without specific optimizations.
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= c_x_size * c_y_size * c_z_size) return;

    const size_t it_z = tid % c_z_size;
    const size_t it_y = (tid / c_z_size) % c_y_size;
    const size_t it_x = (tid / c_z_size) / c_y_size;

    const float cx = c_origin_x + it_x * c_dx;
    const float cy = c_origin_y + it_y * c_dy;
    const float cz = c_origin_z + it_z * c_dz;

    float local_density = 0.0f;

    const size_t cell_x = (cx - c_origin_x) / c_cx;
    const size_t cell_y = (cy - c_origin_y) / c_cy;
    const size_t cell_z = (cz - c_origin_z) / c_cz;

    if (cell_x >= c_nc_x || cell_y >= c_nc_y || cell_z >= c_nc_z) return;

    const size_t cell_idx = cell_x * c_nc_y * c_nc_z + cell_y * c_nc_z + cell_z;

    for (int x = -1; x <= 1; ++x)
    {
        size_t neighbor_x = cell_x + x;
        if (neighbor_x >= c_nc_x) continue;

        for (int y = -1; y <= 1; ++y)
        {
            size_t neighbor_y = cell_y + y;
            if (neighbor_y >= c_nc_y) continue;
            for (int z = -1; z <= 1; ++z)
            {
                size_t neighbor_z = cell_z + z;
                if (neighbor_z >= c_nc_z) continue;

                size_t neighbor_id = neighbor_x * c_nc_y * c_nc_z +
                                    neighbor_y * c_nc_z +
                                    neighbor_z;

                size_t occupation = occupations[neighbor_id];

                for (size_t atom_idx = 0; atom_idx < occupation; ++atom_idx) {
                    const float4 atom = partitioned_molecule[neighbor_id * max_occupation + atom_idx];

                    const float delta_x = cx - atom.x;
                    const float delta_y = cy - atom.y;
                    const float delta_z = cz - atom.z;

                    const float distance_squared = delta_x * delta_x
                                                 + delta_y * delta_y
                                                 + delta_z * delta_z;
                    if (distance_squared < 144.0f)
                    local_density += slater_densities[(int)(atom.w)-1](sqrt(distance_squared), 0.0f, 0.0f);
                    // Removed unnecessary printf statement
                }
            }
        }
    }

    density_field[tid] = local_density;
}


__global__ void _gpu_density_field_relevant_neighborhood(
    float* __restrict__ density_field,
    const float4* partitioned_molecule,
    size_t* occupations,
    const size_t max_occupation
)
{
    // The simplest approach is to use just the thread id to identify the point. From
    // that we access the complete neighborhood

    // This first versions serves as a way of undestanding how well the algorithm behaves without specific optimizations.
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= c_x_size * c_y_size * c_z_size) return;

    const size_t it_z = tid % c_z_size;
    const size_t it_y = (tid / c_z_size) % c_y_size;
    const size_t it_x = (tid / c_z_size) / c_y_size;

    const float cx = c_origin_x + it_x * c_dx;
    const float cy = c_origin_y + it_y * c_dy;
    const float cz = c_origin_z + it_z * c_dz;

    const float px = cx;
    const float py = cy;
    const float pz = cz;

    float local_density = 0.0f;

    const size_t cell_x = (cx - c_origin_x) / c_cx;
    const size_t cell_y = (cy - c_origin_y) / c_cy;
    const size_t cell_z = (cz - c_origin_z) / c_cz;

    if (cell_x >= c_nc_x || cell_y >= c_nc_y || cell_z >= c_nc_z) return;

    const size_t cell_idx = cell_x * c_nc_y * c_nc_z + cell_y * c_nc_z + cell_z;

    const float dx = px - c_origin_x - cell_x * c_cx;
    const float dy = py - c_origin_y - cell_y * c_cy;
    const float dz = pz - c_origin_z - cell_z * c_cz;

    const float dx_p1 = dx - c_cx;
    const float dy_p1 = dy - c_cy;
    const float dz_p1 = dz - c_cz;

    const float dx_m1 = dx + c_cx;
    const float dy_m1 = dy + c_cy;
    const float dz_m1 = dz + c_cz;

    const float arr[] = {
        dx_m1 * dx_m1 + dy_m1 * dy_m1 + dz_m1 * dz_m1, dx_m1 * dx_m1 + dy_m1 * dy_m1, dx_m1 * dx_m1 + dy_m1 * dy_m1 + dz_p1 * dz_p1,
        dx_m1 * dx_m1 + dz_m1 * dz_m1, dx_m1 * dx_m1, dx_m1 * dx_m1 + dz_p1 * dz_p1,
        dx_m1 * dx_m1 + dy_p1 * dy_p1 + dz_m1 * dz_m1, dx_m1 * dx_m1 + dy_p1 * dy_p1, dx_m1 * dx_m1 + dy_p1 * dy_p1 + dz_p1 * dz_p1,
        dy_m1 * dy_m1 + dz_m1 * dz_m1, dy_m1 * dy_m1, dy_m1 * dy_m1 + dz_p1 * dz_p1,
        dz_m1 * dz_m1, 0.0f, dz_p1 * dz_p1,
        dy_p1 * dy_p1 + dz_m1 * dz_m1, dy_p1 * dy_p1, dy_p1 * dy_p1 + dz_p1 * dz_p1,
        dx_p1 * dx_p1 + dy_m1 * dy_m1 + dz_m1 * dz_m1, dx_p1 * dx_p1 + dy_m1 * dy_m1, dx_p1 * dx_p1 + dy_m1 * dy_m1 + dz_p1 * dz_p1,
        dx_p1 * dx_p1 + dz_m1 * dz_m1, dx_p1 * dx_p1, dx_p1 * dx_p1 + dz_p1 * dz_p1,
        dx_p1 * dx_p1 + dy_p1 * dy_p1 + dz_m1 * dz_m1, dx_p1 * dx_p1 + dy_p1 * dy_p1, dx_p1 * dx_p1 + dy_p1 * dy_p1 + dz_p1 * dz_p1
    };

    for (int x = -1; x <= 1; ++x)
    {
        size_t neighbor_x = cell_x + x;
        if (neighbor_x >= c_nc_x) continue;

        for (int y = -1; y <= 1; ++y)
        {
            size_t neighbor_y = cell_y + y;
            if (neighbor_y >= c_nc_y) continue;
            for (int z = -1; z <= 1; ++z)
            {
                size_t neighbor_z = cell_z + z;
                if (neighbor_z >= c_nc_z) continue;

                size_t absolute_neighbor_id = neighbor_x * c_nc_y * c_nc_z +
                                    neighbor_y * c_nc_z +
                                    neighbor_z;

                size_t relative_neighbor_id = (x + 1) * 9 + (y + 1) * 3 + (z + 1);
                if (arr[relative_neighbor_id] < 144.0f)
                {
                    size_t occupation = occupations[absolute_neighbor_id];

                    for (size_t atom_idx = 0; atom_idx < occupation; ++atom_idx) {
                        const float4 atom = partitioned_molecule[absolute_neighbor_id * max_occupation + atom_idx];

                        const float delta_x = cx - atom.x;
                        const float delta_y = cy - atom.y;
                        const float delta_z = cz - atom.z;

                        const float distance_squared = delta_x * delta_x
                        + delta_y * delta_y
                        + delta_z * delta_z;
                        if (distance_squared < 144.0f)
                            local_density += slater_densities[(int)(atom.w)-1](sqrt(distance_squared), 0.0f, 0.0f);
                        // Removed unnecessary printf statement
                    }
                }
            }
        }
    }

    density_field[tid] = local_density;
}

__global__ void _gpu_density_field_atom_if_else(
    float* __restrict__ density_field,
    const float dx,
    const float dy,
    const float dz,
    const size_t x_size,
    const size_t y_size,
    const size_t z_size,
    const float4* atoms,
    const size_t n_atoms
)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t points = x_size * y_size * z_size;
    const float radius = 12.0f;


    for (size_t atom_id = tid; atom_id < n_atoms; atom_id += blockDim.x * gridDim.x)
    {
        const float4 atom = atoms[atom_id];

        const float starting_x = c_origin_x + floor((atom.x - radius - c_origin_x) / dx) * dx;
        const float starting_y = c_origin_y + floor((atom.y - radius - c_origin_y) / dy) * dy;
        const float starting_z = c_origin_z + floor((atom.z - radius - c_origin_z) / dz) * dz;

        const float end_x = min(c_origin_x + x_size * dx, starting_x + 2 * radius);
        const float end_y = min(c_origin_y + y_size * dy, starting_y + 2 * radius);
        const float end_z = min(c_origin_z + z_size * dz, starting_z + 2* radius);

        for (float x = starting_x; x <= end_x; x += dx)
        {
            for (float y = starting_y; y <= end_y; y += dy)
            {
            for (float z = starting_z; z <= end_z; z += dz)
            {
                const float delta_x = x - atom.x;
                const float delta_y = y - atom.y;
                const float delta_z = z - atom.z;

                const float distance_squared = delta_x * delta_x
                            + delta_y * delta_y
                            + delta_z * delta_z;

                if (distance_squared < radius * radius)
                {
                    const size_t idx_x = (x - c_origin_x) / dx;
                    const size_t idx_y = (y - c_origin_y) / dy;
                    const size_t idx_z = (z - c_origin_z) / dz;

                    const size_t idx = idx_x * y_size * z_size + idx_y * z_size + idx_z;

                    atomicAdd(&density_field[idx], slater_functions((int)atom.w, sqrt(distance_squared), 0.0f, 0.0f));
                }
            }
            }
        }

    }
}


__global__ void _gpu_density_field_atom_func_ptr(
    float* __restrict__ density_field,
    const float dx,
    const float dy,
    const float dz,
    const size_t x_size,
    const size_t y_size,
    const size_t z_size,
    const float4* atoms,
    const size_t n_atoms
)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t points = x_size * y_size * z_size;
    const float radius = 12.0f;

    if (tid >= n_atoms) return;

    size_t atom_id = tid;

    // for (size_t atom_id = tid; atom_id < n_atoms; atom_id += blockDim.x * gridDim.x)
    // {
        const float4 atom = atoms[tid];

        const float starting_x = c_origin_x + floor((atom.x - radius - c_origin_x) / dx) * dx;
        const float starting_y = c_origin_y + floor((atom.y - radius - c_origin_y) / dy) * dy;
        const float starting_z = c_origin_z + floor((atom.z - radius - c_origin_z) / dz) * dz;

        const float end_x = min(c_origin_x + x_size * dx, starting_x + 2 * radius);
        const float end_y = min(c_origin_y + y_size * dy, starting_y + 2 * radius);
        const float end_z = min(c_origin_z + z_size * dz, starting_z + 2* radius);

        const slater_density density = slater_densities[(int)(atoms[atom_id].w)-1];

        for (float x = starting_x; x <= end_x; x += dx)
        {
            for (float y = starting_y; y <= end_y; y += dy)
            {
            for (float z = starting_z; z <= end_z; z += dz)
            {
                const float delta_x = x - atom.x;
                const float delta_y = y - atom.y;
                const float delta_z = z - atom.z;

                const float distance_squared = delta_x * delta_x
                            + delta_y * delta_y
                            + delta_z * delta_z;

                if (distance_squared < radius * radius)
                {
                    const size_t idx_x = (x - c_origin_x) / dx;
                    const size_t idx_y = (y - c_origin_y) / dy;
                    const size_t idx_z = (z - c_origin_z) / dz;

                    const size_t idx = idx_x * y_size * z_size + idx_y * z_size + idx_z;

                    atomicAdd(&density_field[idx], density(sqrtf(distance_squared), 0.0f, 0.0f));
                }
            }
            }
        }

    // }
}

__global__ void _gpu_density_field_if_else_shared(
    float* __restrict__ density_field,
    const float dx,
    const float dy,
    const float dz,
    const size_t x_size,
    const size_t y_size,
    const size_t z_size,
    const float4* atoms,
    const size_t n_atoms
)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t points = c_x_size * c_y_size * c_z_size;
    const float max_distance = 144.0f;

    const int atoms_in_shared = 512;
    __shared__ float4 local_atom[atoms_in_shared];
    int batches = (n_atoms + atoms_in_shared - 1) / atoms_in_shared;

    for (int atom_batch = 0; atom_batch < batches; ++atom_batch)
    {
        __syncthreads();
        // preload atoms in shared memory
        for (size_t local_idx = threadIdx.x; local_idx < atoms_in_shared && atom_batch * atoms_in_shared + local_idx < n_atoms; local_idx += blockDim.x)
        {
            local_atom[local_idx] = atoms[atom_batch * atoms_in_shared + local_idx];
        }
        __syncthreads();

        for (size_t i = tid; i < points; i += blockDim.x * gridDim.x) {
            const size_t it_z = i % c_z_size;
            const size_t it_y = (i / c_z_size) % c_y_size;
            const size_t it_x = (i / c_z_size) / c_y_size;

            const float cx = c_origin_x + it_x * c_dx;
            const float cy = c_origin_y + it_y * c_dy;
            const float cz = c_origin_z + it_z * c_dz;
            float local_density = 0.0f;

            for (size_t atom_idx = 0; atom_idx < atoms_in_shared; ++atom_idx)
            {
                if (atoms_in_shared * atom_batch + atom_idx < n_atoms)
                {

                    const float delta_x = cx - local_atom[atom_idx].x;
                    const float delta_y = cy - local_atom[atom_idx].y;
                    const float delta_z = cz - local_atom[atom_idx].z;

                    const float distance_squared = delta_x * delta_x
                                                + delta_y * delta_y
                                                + delta_z * delta_z;
                    if (distance_squared < max_distance)
                    {
                        local_density += slater_functions((int) local_atom[atom_idx].w, sqrt(distance_squared), 0.0f, 0.0f);
                    }
                }
            }
            density_field[i] += local_density;
        }
    }
}

__global__ void _gpu_density_field_func_ptr_shared(
    float* __restrict__ density_field,
    const float dx,
    const float dy,
    const float dz,
    const size_t x_size,
    const size_t y_size,
    const size_t z_size,
    const float4* atoms,
    const size_t n_atoms
)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    size_t points = x_size * y_size * z_size;
    const float max_distance = 144.0f;

    const int atoms_in_shared = 512;
    __shared__ float4 local_atom[atoms_in_shared];
    int batches = (n_atoms + atoms_in_shared - 1) / atoms_in_shared;

    for (int atom_batch = 0; atom_batch < batches; ++atom_batch)
    {
        __syncthreads();
        // preload atoms in shared memory
        for (size_t local_idx = threadIdx.x; local_idx < atoms_in_shared && local_idx + atom_batch * atoms_in_shared < n_atoms; local_idx += blockDim.x)
        {
            if (local_idx < n_atoms)
                local_atom[local_idx] = atoms[atom_batch * atoms_in_shared + local_idx];
        }
        __syncthreads();

        // for (size_t i = tid; i < points; i += blockDim.x * gridDim.x) {
            const size_t it_z = tid % z_size;
            const size_t it_y = (tid / z_size) % y_size;
            const size_t it_x = (tid / z_size) / y_size;

            const float cx = c_origin_x + it_x * dx;
            const float cy = c_origin_y + it_y * dy;
            const float cz = c_origin_z + it_z * dz;
            float local_density = 0.0f;
            for (size_t atom_idx = 0; atom_idx < atoms_in_shared && atom_idx + atom_batch * atoms_in_shared < n_atoms; ++atom_idx)
            {
                if (atom_idx < n_atoms) {
                    const float delta_x = cx - local_atom[atom_idx].x;
                    const float delta_y = cy - local_atom[atom_idx].y;
                    const float delta_z = cz - local_atom[atom_idx].z;

                    const float distance_squared = delta_x * delta_x
                                                + delta_y * delta_y
                                                + delta_z * delta_z;
                    if (distance_squared < max_distance)
                    {
                        local_density += slater_densities[(int)(local_atom[atom_idx].w)-1](sqrtf(distance_squared), 0.0f, 0.0f);
                    }
                }
            }
            density_field[tid] += local_density;
        // }

    }
}


__global__ void _gpu_print_cells(float4* cells, size_t* occupations, size_t max_occupation, size_t n_cells)
{
    for (size_t i = 0; i < n_cells; ++i)
    {
        if (occupations[i] > 0) {
            printf("Cell[%lu]. %lu atoms\n", i, occupations[i]);
        }
    }
}


// ============================ ALGORITHM WITH EMPTY NEIGHBORHOOD PREPROCESSING ============================

__global__ void _gpu_exclude_empty_neighborhoods(
    size_t* included_neighbors,
    unsigned int* n_included_neighbors,
    size_t* d_occupations,
    size_t c_n_atoms
)
{
    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid >= c_nc_x * c_nc_y * c_nc_z) return;

    size_t cell_z = tid % c_nc_z;
    size_t cell_y = (tid / c_nc_z) % c_nc_y;
    size_t cell_x = (tid / (c_nc_y * c_nc_z));

    size_t total_atoms_in_neighborhood = 0;

    for (int x = -1; x <= 1; ++x)
    {
        for (int y = -1; y <= 1; ++y)
        {
            for (int z = -1; z <= 1; ++z)
            {
                size_t neighbor_x = cell_x + x;
                size_t neighbor_y = cell_y + y;
                size_t neighbor_z = cell_z + z;

                if (neighbor_x < c_nc_x && neighbor_y < c_nc_y && neighbor_z < c_nc_z) {
                    size_t neighbor_id = neighbor_x * c_nc_y * c_nc_z +
                                         neighbor_y * c_nc_z +
                                         neighbor_z;

                    total_atoms_in_neighborhood += d_occupations[neighbor_id];
                }
            }
        }
    }

    if (total_atoms_in_neighborhood != 0)
    {
        size_t insert_pos = atomicAdd(n_included_neighbors, 1);
        included_neighbors[insert_pos] = tid;
    }
}

__global__ void _gpu_density_field_excluded_neighborhood(
    float* __restrict__ density_field,
    const float4* partitioned_molecule,
    const size_t* included_bins,
    const size_t ppb_x,
    const size_t ppb_y,
    const size_t ppb_z,
    const unsigned int* n_included_bins,
    const size_t max_occupation,
    const size_t* occupations
)
{
    
    float local_density = 0.0f;
    
    
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;

    // Now wrap the increasing thread Ids into the cubes

    const size_t mapped_bin_offset = tid / (ppb_x * ppb_y * ppb_z);
    const size_t point_idx = tid % (ppb_x * ppb_y * ppb_z);

    const size_t wrapped_k = point_idx % ppb_z;
    const size_t wrapped_j = (point_idx / ppb_z) % ppb_y;
    const size_t wrapped_i = (point_idx / (ppb_z * ppb_y));

    const size_t mapped_bin = included_bins[mapped_bin_offset];

    if (mapped_bin_offset >= *n_included_bins) return;

    // printf("Wrapping: tid %lu -> i = %lu, j = %lu, k = %lu offset %lu - bin %lu\n", tid, wrapped_i, wrapped_j, wrapped_k,
    //     mapped_bin_offset, mapped_bin);

    const size_t bin_z = mapped_bin % c_nc_z;
    const size_t bin_y = (mapped_bin / c_nc_z) % c_nc_y;
    const size_t bin_x = (mapped_bin / (c_nc_y * c_nc_z));

    const size_t final_z = wrapped_k + (bin_z/*+1*/) * ppb_z;
    const size_t final_y = wrapped_j + (bin_y/*+1*/) * ppb_y;
    const size_t final_x = wrapped_i + (bin_x/*+1*/) * ppb_x;

    // if (final_x >= c_x_size || final_y >= c_y_size || final_z >= c_z_size) return;

    const size_t final_idx = final_x * c_y_size * c_z_size + final_y * c_z_size + final_z;

    const float p_x = c_origin_x + final_x * c_dx;
    const float p_y = c_origin_y + final_y * c_dy;
    const float p_z = c_origin_z + final_z * c_dz;

    for (int x = -1; x <= 1; ++x)
    {
        size_t neighbor_x = bin_x + x;
        if (neighbor_x >= c_nc_x) continue;

        for (int y = -1; y <= 1; ++y)
        {
            size_t neighbor_y = bin_y + y;
            if (neighbor_y >= c_nc_y) continue;
            for (int z = -1; z <= 1; ++z)
            {
                size_t neighbor_z = bin_z + z;
                if (neighbor_z >= c_nc_z) continue;

                size_t neighbor_id = neighbor_x * c_nc_y * c_nc_z +
                                    neighbor_y * c_nc_z +
                                    neighbor_z;

                size_t occupation = occupations[neighbor_id];

                for (size_t atom_idx = 0; atom_idx < occupation; ++atom_idx) {
                    const float4 atom = partitioned_molecule[neighbor_id * max_occupation + atom_idx];

                    const float delta_x = p_x - atom.x;
                    const float delta_y = p_y - atom.y;
                    const float delta_z = p_z - atom.z;

                    const float distance_squared = delta_x * delta_x
                                                 + delta_y * delta_y
                                                 + delta_z * delta_z;
                    if (distance_squared < 144.0f)
                    local_density += slater_densities[(int)(atom.w)-1](sqrt(distance_squared), 0.0f, 0.0f);
                    // Removed unnecessary printf statement
                }
            }
        }
    }

    density_field[final_idx] = local_density;
}

__global__ void _gpu_density_field_excluded_neighborhood_ONE_BLOCK_PER_BIN(
    float* __restrict__ density_field,
    const float4* __restrict__ partitioned_molecule,
    const size_t* __restrict__ included_bins,
    const size_t ppb_x,
    const size_t ppb_y,
    const size_t ppb_z,
    const unsigned int* __restrict__ n_included_bins,
    const size_t max_occupation,
    const size_t* __restrict__ occupations
)
{
    
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t local_tid = threadIdx.x;
    size_t block_idx = blockIdx.x;

    const size_t mapped_bin_offset = block_idx;
    const size_t points_per_bin = ppb_x * ppb_y * ppb_z;
    const size_t mapped_bin = included_bins[mapped_bin_offset];

    const size_t bin_z = mapped_bin % c_nc_z;
    const size_t bin_y = (mapped_bin / c_nc_z) % c_nc_y;
    const size_t bin_x = (mapped_bin / (c_nc_y * c_nc_z));

    extern __shared__ float4 atom_local[];
    size_t atoms_loaded = 0;

    // if (mapped_bin_offset >= *n_included_bins) return;

    // We start by preloading the atomic information into shared memory
    #pragma unroll
    for (int x = -1; x <= 1; ++x)
    {
        size_t neighbor_x = bin_x + x;
        if (neighbor_x >= c_nc_x) continue;

        #pragma unroll
        for (int y = -1; y <= 1; ++y)
        {
            size_t neighbor_y = bin_y + y;
            if (neighbor_y >= c_nc_y) continue;
        
            #pragma unroll
            for (int z = -1; z <= 1; ++z)
            {
                size_t neighbor_z = bin_z + z;
                if (neighbor_z >= c_nc_z) continue;

                size_t neighbor_id = neighbor_x * c_nc_y * c_nc_z +
                                    neighbor_y * c_nc_z +
                                    neighbor_z;

                size_t occupation = occupations[neighbor_id];
                
                
                for (size_t loaded_atom_id = local_tid; loaded_atom_id < occupation; loaded_atom_id += blockDim.x)
                {
                    size_t position = loaded_atom_id + atoms_loaded;
                    atom_local[position] = partitioned_molecule[neighbor_id * max_occupation + loaded_atom_id];
                }
                atoms_loaded += occupation;
                
            }
        }
    }
    __syncthreads();

    // And then do the necessary operations
    for (size_t idx = local_tid; idx < points_per_bin; idx += blockDim.x)
    {
        float local_density = 0.0f;
        const size_t wrapped_k = idx % ppb_z;
        const size_t wrapped_j = (idx / ppb_z) % ppb_y;
        const size_t wrapped_i = idx / (ppb_y * ppb_z);

        const size_t final_z = wrapped_k + bin_z * ppb_z;
        const size_t final_y = wrapped_j + bin_y * ppb_y;
        const size_t final_x = wrapped_i + bin_x * ppb_x;

        // if (final_x >= c_x_size || final_y >= c_y_size || final_z >= c_z_size) return;

        const size_t final_idx = final_x * c_y_size * c_z_size + final_y * c_z_size + final_z;

        const float p_x = c_origin_x + final_x * c_dx;
        const float p_y = c_origin_y + final_y * c_dy;
        const float p_z = c_origin_z + final_z * c_dz;

        for (size_t atom_idx = 0; atom_idx < atoms_loaded; ++atom_idx) {
            const float4 atom = atom_local[atom_idx];

            const float delta_x = p_x - atom.x;
            const float delta_y = p_y - atom.y;
            const float delta_z = p_z - atom.z;


            const float distance_squared = delta_x * delta_x
                                        + delta_y * delta_y
                                        + delta_z * delta_z;
            if (distance_squared < 144.0f)
            local_density += slater_densities[(int)(atom.w)-1](sqrt(distance_squared), 0.0f, 0.0f);
            // // Removed unnecessary printf statement
        }
        density_field[final_idx] = local_density;
    }
}


__global__ void _gpu_density_field_excluded_neighborhood_ONE_BLOCK_PER_POINT(
    float* __restrict__ density_field,
    const float4* __restrict__ partitioned_molecule,
    const size_t* __restrict__ included_bins,
    const size_t ppb_x,
    const size_t ppb_y,
    const size_t ppb_z,
    const unsigned int* __restrict__ n_included_bins,
    const size_t max_occupation,
    const size_t* __restrict__ occupations
)
{
    
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t local_tid = threadIdx.x;

    const size_t points_per_bin = ppb_x * ppb_y * ppb_z;
    const size_t mapped_bin_offset = blockIdx.x / points_per_bin;
    const size_t mapped_bin = included_bins[mapped_bin_offset];

    const size_t bin_z = mapped_bin % c_nc_z;
    const size_t bin_y = (mapped_bin / c_nc_z) % c_nc_y;
    const size_t bin_x = (mapped_bin / (c_nc_y * c_nc_z));

    extern __shared__ float shared_mem[];

    float4* atom_local = (float4*)shared_mem;
    float* threads_density = (float*)(shared_mem + 1024 * 4);

    size_t atoms_loaded = 0;

    if (mapped_bin_offset >= *n_included_bins) return;

    threads_density[local_tid] = 0.0f;

    size_t idx = blockIdx.x % points_per_bin;

    const size_t wrapped_k = idx % ppb_z;
    const size_t wrapped_j = (idx / ppb_z) % ppb_y;
    const size_t wrapped_i = idx / (ppb_y * ppb_z);

    const size_t final_z = wrapped_k + bin_z * ppb_z;
    const size_t final_y = wrapped_j + bin_y * ppb_y;
    const size_t final_x = wrapped_i + bin_x * ppb_x;

    
    if (final_x >= c_x_size || final_y >= c_y_size || final_z >= c_z_size) return;
    
    const size_t final_idx = final_x * c_y_size * c_z_size + final_y * c_z_size + final_z;
    // if (threadIdx.x == 0)
    //     printf("final_idx: %lu\n", final_idx);

    const float p_x = c_origin_x + final_x * c_dx;
    const float p_y = c_origin_y + final_y * c_dy;
    const float p_z = c_origin_z + final_z * c_dz;

    // threads_density[local_tid] = 0.0f;

    if (local_tid == 0) {
        density_field[final_idx] = 2.0f;
        // density_field[final_idx] = threads_density[0];
    }

}

__global__ void _gpu_density_field_excluded_neighborhood_ONE_BLOCK_PER_BIN_OPT(
    float* __restrict__ density_field,
    const float4* __restrict__ partitioned_molecule,
    const size_t* __restrict__ included_bins,
    const size_t ppb_x,
    const size_t ppb_y,
    const size_t ppb_z,
    const unsigned int* __restrict__ n_included_bins,
    const size_t max_occupation,
    const size_t* __restrict__ occupations
) {
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    size_t local_tid = threadIdx.x;
    size_t block_idx = blockIdx.x;

    const size_t mapped_bin_offset = block_idx;
    const size_t points_per_bin = ppb_x * ppb_y * ppb_z;

    // Bounds check for safety
    if (mapped_bin_offset >= *n_included_bins) return;

    const size_t mapped_bin = included_bins[mapped_bin_offset];

    const size_t bin_z = mapped_bin % c_nc_z;
    const size_t bin_y = (mapped_bin / c_nc_z) % c_nc_y;
    const size_t bin_x = (mapped_bin / (c_nc_y * c_nc_z));

    extern __shared__ int memory[];

    float4* atom_local = reinterpret_cast<float4*>(memory);
    size_t* neighbor_occupation = reinterpret_cast<size_t*>(atom_local + 2048);
    size_t* neighbor_occupation_scan = reinterpret_cast<size_t*>(neighbor_occupation + 27);

    // Initialize shared memory arrays
    for (int i = local_tid; i < 27; i += blockDim.x) {
        neighbor_occupation[i] = 0;
        neighbor_occupation_scan[i] = 0;
    }
    __syncthreads();

    // Load atoms from neighboring bins
    size_t atoms_loaded = 0;
    int count = 0;

    #pragma unroll
    for (int x = -1; x <= 1; ++x) {
        size_t neighbor_x = bin_x + x;
        if (neighbor_x >= c_nc_x) continue;

        #pragma unroll
        for (int y = -1; y <= 1; ++y) {
            size_t neighbor_y = bin_y + y;
            if (neighbor_y >= c_nc_y) continue;

            #pragma unroll
            for (int z = -1; z <= 1; ++z) {
                size_t neighbor_z = bin_z + z;
                if (neighbor_z >= c_nc_z) continue;

                size_t neighbor_id = neighbor_x * c_nc_y * c_nc_z +
                                     neighbor_y * c_nc_z +
                                     neighbor_z;

                size_t occupation = occupations[neighbor_id];

                if (threadIdx.x == 0) {
                    neighbor_occupation[count] = occupation;
                }

                for (size_t loaded_atom_id = local_tid; loaded_atom_id < occupation; loaded_atom_id += blockDim.x) {
                    size_t position = loaded_atom_id + atoms_loaded;
                    atom_local[position] = partitioned_molecule[neighbor_id * max_occupation + loaded_atom_id];
                }

                if (threadIdx.x == 0 && count > 0) {
                    neighbor_occupation_scan[count] = neighbor_occupation_scan[count - 1] + neighbor_occupation[count - 1];
                }

                atoms_loaded += occupation;
                count++;
            }
        }
    }

    __syncthreads();

    for (size_t idx = local_tid; idx < points_per_bin; idx += blockDim.x) {
        float local_density = 0.0f;

        const size_t wrapped_k = idx % ppb_z;
        const size_t wrapped_j = (idx / ppb_z) % ppb_y;
        const size_t wrapped_i = idx / (ppb_y * ppb_z);

        const size_t final_z = wrapped_k + bin_z * ppb_z;
        const size_t final_y = wrapped_j + bin_y * ppb_y;
        const size_t final_x = wrapped_i + bin_x * ppb_x;

        const size_t final_idx = final_x * c_y_size * c_z_size + final_y * c_z_size + final_z;

        const float p_x = c_origin_x + final_x * c_dx;
        const float p_y = c_origin_y + final_y * c_dy;
        const float p_z = c_origin_z + final_z * c_dz;

        const float dx = p_x - c_origin_x - bin_x * c_cx;
        const float dy = p_y - c_origin_y - bin_y * c_cy;
        const float dz = p_z - c_origin_z - bin_z * c_cz;

        const float dx_p1 = dx - c_cx;
        const float dy_p1 = dy - c_cy;
        const float dz_p1 = dz - c_cz;

        const float dx_m1 = dx + c_cx;
        const float dy_m1 = dy + c_cy;
        const float dz_m1 = dz + c_cz;

        const float arr[] = {
            dx_m1 * dx_m1 + dy_m1 * dy_m1 + dz_m1 * dz_m1, dx_m1 * dx_m1 + dy_m1 * dy_m1, dx_m1 * dx_m1 + dy_m1 * dy_m1 + dz_p1 * dz_p1,
            dx_m1 * dx_m1 + dz_m1 * dz_m1, dx_m1 * dx_m1, dx_m1 * dx_m1 + dz_p1 * dz_p1,
            dx_m1 * dx_m1 + dy_p1 * dy_p1 + dz_m1 * dz_m1, dx_m1 * dx_m1 + dy_p1 * dy_p1, dx_m1 * dx_m1 + dy_p1 * dy_p1 + dz_p1 * dz_p1,
            dy_m1 * dy_m1 + dz_m1 * dz_m1, dy_m1 * dy_m1, dy_m1 * dy_m1 + dz_p1 * dz_p1,
            dz_m1 * dz_m1, 0.0f, dz_p1 * dz_p1,
            dy_p1 * dy_p1 + dz_m1 * dz_m1, dy_p1 * dy_p1, dy_p1 * dy_p1 + dz_p1 * dz_p1,
            dx_p1 * dx_p1 + dy_m1 * dy_m1 + dz_m1 * dz_m1, dx_p1 * dx_p1 + dy_m1 * dy_m1, dx_p1 * dx_p1 + dy_m1 * dy_m1 + dz_p1 * dz_p1,
            dx_p1 * dx_p1 + dz_m1 * dz_m1, dx_p1 * dx_p1, dx_p1 * dx_p1 + dz_p1 * dz_p1,
            dx_p1 * dx_p1 + dy_p1 * dy_p1 + dz_m1 * dz_m1, dx_p1 * dx_p1 + dy_p1 * dy_p1, dx_p1 * dx_p1 + dy_p1 * dy_p1 + dz_p1 * dz_p1
        };

        for (size_t neighbor = 0; neighbor < 27; ++neighbor) {
            if (arr[neighbor] > 150.0f)
                continue;

            for (size_t atom_idx = 0; atom_idx < neighbor_occupation[neighbor]; ++atom_idx) {
                float4 atom = atom_local[atom_idx + neighbor_occupation_scan[neighbor]];

                float dx = p_x - atom.x;
                float dy = p_y - atom.y;
                float dz = p_z - atom.z;

                float dist2 = dx * dx + dy * dy + dz * dz;

                if (dist2 < 144.0f) {
                    local_density += slater_densities[(int)(atom.w) - 1](sqrtf(dist2), 0.0f, 0.0f);
                }
            }
        }

        density_field[final_idx] = local_density;
    }
}


#endif // __EDDI_GPU_DENSITY_FIELD_KERNEL_CUH__