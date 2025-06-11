#include "eddi_gpu_density_field.h"
#include "eddi_gpu_density_field_kernel.cuh"





void _load_origin_in_constant_memory(const eddi_point_t* origin)
{

    check_error(cudaMemcpyToSymbol(c_origin_x, &(origin->x), sizeof(float)), "cuda memcpy to symbol - ox");
    check_error(cudaMemcpyToSymbol(c_origin_y, &(origin->y), sizeof(float)), "cuda memcpy to symbol - oy");
    check_error(cudaMemcpyToSymbol(c_origin_z, &(origin->z), sizeof(float)), "cuda memcpy to symbol - oz");

}

void _load_field_data_in_constant_memory(
    const eddi_size_t x_size,
    const eddi_size_t y_size,
    const eddi_size_t z_size,
    const eddi_real_t dx,
    const eddi_real_t dy,
    const eddi_real_t dz
)
{
    check_error(cudaMemcpyToSymbol(c_x_size, &x_size, sizeof(size_t)), "cuda memcpy to symbol - x_size");
    check_error(cudaMemcpyToSymbol(c_y_size, &y_size, sizeof(size_t)), "cuda memcpy to symbol - y_size");
    check_error(cudaMemcpyToSymbol(c_z_size, &z_size, sizeof(size_t)), "cuda memcpy to symbol - z_size");

    check_error(cudaMemcpyToSymbol(c_dx, &dx, sizeof(float)), "cuda memcpy to symbol - dx");
    check_error(cudaMemcpyToSymbol(c_dy, &dy, sizeof(float)), "cuda memcpy to symbol - dy");
    check_error(cudaMemcpyToSymbol(c_dz, &dz, sizeof(float)), "cuda memcpy to symbol - dz");
}

void _load_cell_list_data_in_constant_memory(
    const eddi_size_t nc_x,
    const eddi_size_t nc_y,
    const eddi_size_t nc_z,
    const eddi_real_t cx,
    const eddi_real_t cy,
    const eddi_real_t cz
)
{
    check_error(cudaMemcpyToSymbol(c_nc_x, &nc_x, sizeof(size_t)), "cuda memcpy to symbol - nc_x");
    check_error(cudaMemcpyToSymbol(c_nc_y, &nc_y, sizeof(size_t)), "cuda memcpy to symbol - nc_y");
    check_error(cudaMemcpyToSymbol(c_nc_z, &nc_z, sizeof(size_t)), "cuda memcpy to symbol - nc_z");

    check_error(cudaMemcpyToSymbol(c_cx, &cx, sizeof(float)), "cuda memcpy to symbol - cx");
    check_error(cudaMemcpyToSymbol(c_cy, &cy, sizeof(float)), "cuda memcpy to symbol - cy");
    check_error(cudaMemcpyToSymbol(c_cz, &cz, sizeof(float)), "cuda memcpy to symbol - cz");

}


void eddi_gpu_print_origin()
{
    _gpu_print_origin<<<1, 256>>>();
    check_error(cudaDeviceSynchronize(), "cuda device synchronize - gpu print origin");
    check_error(cudaGetLastError(), "cuda get last error - gpu print origin");
}


__global__ void _gpu_access_all_func_ptr()
{
    for (int i = 1; i <= EDDI_GPU_N_SLATER_DENSITIES; ++i)
    {
        printf("%d: %f\n", i, slater_densities[i - 1](0.0, 0.0, 0.0));
    }
}

void eddi_gpu_compute_density_field_POINTS(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule)
{
    printf("==== DENSITY COMPUTATION (POINTS) ====\n");

    // Everything is assumed to be already on the device, so that we just need to call the kernel
    unsigned int threads_per_block = 512;
    unsigned int blocks = (density_field->x_size * density_field->y_size * density_field->z_size + threads_per_block - 1) / threads_per_block;

    printf("Threads per block: %u\n", threads_per_block);
    printf("Number of blocks: %u\n", blocks);
    printf("Density Field Parameters:\n");
    printf("dx: %f, dy: %f, dz: %f\n", density_field->dx, density_field->dy, density_field->dz);
    printf("x_size: %lu, y_size: %lu, z_size: %lu\n", density_field->x_size, density_field->y_size, density_field->z_size);
    printf("Number of atoms: %lu\n", molecule->n_atoms);


    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events

    check_error(cudaEventCreate(&start), "cuda event create - start");
    check_error(cudaEventCreate(&stop), "cuda event create - stop");

    // Record start event
    check_error(cudaEventRecord(start, 0), "cuda event record - start");

    _gpu_density_field_func_ptr_shared<<<blocks, threads_per_block>>>(
        density_field->field,
        density_field->dx,
        density_field->dy,
        density_field->dz,
        density_field->x_size,
        density_field->y_size,
        density_field->z_size,
        molecule->atoms,
        molecule->n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop, 0), "cuda event record - stop");
    check_error(cudaEventSynchronize(stop), "cuda event synchronize - stop");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime, start, stop), "cuda event elapsed time");
    printf("Point-Based Kernel execution time: %f ms\n", elapsedTime);

    printf("FINAL KERNEL TIME: %f ms\n", elapsedTime);
    // Free allocated memory
}


void eddi_gpu_compute_density_field_ATOMS(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule)
{
    printf("==== DENSITY COMPUTATION (ATOMS) ====\n");

    unsigned int threads_per_block = 512;
    unsigned int blocks = (molecule->n_atoms + threads_per_block - 1) / threads_per_block;
    // Everything is assumed to be already on the device, so that we just need to call the kernel
    printf("Threads per block: %u\n", threads_per_block);
    printf("Number of blocks: %u\n", blocks);
    printf("Density Field Parameters:\n");
    printf("dx: %f, dy: %f, dz: %f\n", density_field->dx, density_field->dy, density_field->dz);
    printf("x_size: %lu, y_size: %lu, z_size: %lu\n", density_field->x_size, density_field->y_size, density_field->z_size);
    printf("Number of atoms: %lu\n", molecule->n_atoms);
    
    
    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events
    check_error(cudaEventCreate(&start), "cuda event create - start");
    check_error(cudaEventCreate(&stop), "cuda event create - stop");

    // Record start event
    check_error(cudaEventRecord(start, 0), "cuda event record - start");
    

    _gpu_density_field_atom_func_ptr<<<blocks, threads_per_block>>>(
        density_field->field,
        density_field->dx,
        density_field->dy,
        density_field->dz,
        density_field->x_size,
        density_field->y_size,
        density_field->z_size,
        molecule->atoms,
        molecule->n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop, 0), "cuda event record - stop");
    check_error(cudaEventSynchronize(stop), "cuda event synchronize - stop");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime, start, stop), "cuda event elapsed time");
    printf("Atom-Based Kernel execution time: %f ms\n", elapsedTime);

    // Destroy events
    check_error(cudaEventDestroy(start), "cuda event destroy - start");
    check_error(cudaEventDestroy(stop), "cuda event destroy - stop");

    check_error(cudaDeviceSynchronize(), "cuda device synchronize - compute density if else");
    check_error(cudaGetLastError(), "cuda get last error");

    printf("FINAL KERNEL TIME: %f ms\n", elapsedTime);
    // Free allocated memory
}

void eddi_gpu_compute_density_field_CELL_LIST(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info)
{
    printf("==== DENSITY COMPUTATION (COMPLETE NEIGHBORHOOD) ====\n");

    const float c_x = info->cx;
    const float c_y = info->cy;
    const float c_z = info->cz;
    const size_t nc_x = (density_field->x_size * density_field->dx) / c_x;
    const size_t nc_y = (density_field->y_size * density_field->dy) / c_y;
    const size_t nc_z = (density_field->z_size * density_field->dz) / c_z;

    const size_t n_points = density_field->x_size * density_field->y_size * density_field->z_size;


    const size_t nc_tot = nc_x * nc_y * nc_z;

    const float multiplier = molecule->n_atoms * 0.5;
    const eddi_size_t c_n_atoms = multiplier + 1;

    size_t threads_per_block = 512;
    size_t blocks = (nc_tot + threads_per_block - 1) / threads_per_block;

    float4* d_cells;
    size_t* d_occupations;

    _load_cell_list_data_in_constant_memory(nc_x, nc_y, nc_z, c_x, c_y, c_z);

    printf("Cell dimensions: cx = %f, cy = %f, cz = %f\n", c_x, c_y, c_z);
    printf("Number of cells along x-axis (nc_x): %lu\n", nc_x);
    printf("Number of cells along y-axis (nc_y): %lu\n", nc_y);
    printf("Number of cells along z-axis (nc_z): %lu\n", nc_z);
    printf("Total number of cells (nc_tot): %lu\n", nc_tot);
    printf("Number of atoms per cell (c_n_atoms): %lu\n", c_n_atoms);
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);

    check_error(cudaMalloc((void**) &d_cells, sizeof(float4) * c_n_atoms * nc_tot), "cuda malloc - cell list");
    check_error(cudaMalloc((void**) &d_occupations, sizeof(size_t) * nc_tot), "cuda malloc - occupations");
    check_error(cudaMemset(d_occupations, 0, sizeof(size_t) * nc_tot), "cuda memset - occupations");

    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events
    check_error(cudaEventCreate(&start), "cuda event create - start");
    check_error(cudaEventCreate(&stop), "cuda event create - stop");

    // Record start event
    check_error(cudaEventRecord(start, 0), "cuda event record - start");

    _gpu_density_partition_atoms<<<blocks, threads_per_block>>>(
        d_cells,
        d_occupations,
        c_n_atoms,
        molecule->atoms,
        molecule->n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop, 0), "cuda event record - stop");
    check_error(cudaEventSynchronize(stop), "cuda event synchronize - stop");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime, start, stop), "cuda event elapsed time");
    printf("Kernel execution time: %f ms\n", elapsedTime);

    // Destroy events
    check_error(cudaEventDestroy(start), "cuda event destroy - start");
    check_error(cudaEventDestroy(stop), "cuda event destroy - stop");

    check_error(cudaDeviceSynchronize(), "cuda device synchronize - partition atoms");
    check_error(cudaGetLastError(), "cuda get last error - partition atoms");

    // Create events
    check_error(cudaEventCreate(&start), "cuda event create - start");
    check_error(cudaEventCreate(&stop), "cuda event create - stop");


    threads_per_block = 512;
    blocks = (n_points + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start_neighborhood, stop_neighborhood;
    float elapsedTime_neighborhood;

    // Create events
    check_error(cudaEventCreate(&start_neighborhood), "cuda event create - start neighborhood");
    check_error(cudaEventCreate(&stop_neighborhood), "cuda event create - stop neighborhood");

    printf("Complete Neighborhood Kernel Configuration:\n");
    printf("Threads per block: %u\n", threads_per_block);
    printf("Number of blocks: %u\n", blocks);
    printf("Total threads: %u\n", threads_per_block * blocks);

    // Record start event
    check_error(cudaEventRecord(start_neighborhood, 0), "cuda event record - start neighborhood");

    _gpu_density_field_complete_neighborhood<<<blocks, threads_per_block>>>(
        density_field->field,
        d_cells,
        d_occupations,
        c_n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop_neighborhood, 0), "cuda event record - stop neighborhood");
    check_error(cudaEventSynchronize(stop_neighborhood), "cuda event synchronize - stop neighborhood");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime_neighborhood, start_neighborhood, stop_neighborhood), "cuda event elapsed time - neighborhood");
    printf("Neighborhood iteration kernel execution time: %f ms\n", elapsedTime_neighborhood);

    // Destroy events
    check_error(cudaEventDestroy(start_neighborhood), "cuda event destroy - start neighborhood");
    check_error(cudaEventDestroy(stop_neighborhood), "cuda event destroy - stop neighborhood");

    check_error(cudaDeviceSynchronize(), "neighborhood iteration - synch");
    check_error(cudaGetLastError(), "neighborhood iteration - get last error");

    // // Create events
    // check_error(cudaEventCreate(&start_neighborhood), "cuda event create - start neighborhood");
    // check_error(cudaEventCreate(&stop_neighborhood), "cuda event create - stop neighborhood");

    // printf("Relevant Neighborhood Kernel Configuration:\n");
    // printf("Threads per block: %u\n", threads_per_block);
    // printf("Number of blocks: %u\n", blocks);
    // printf("Total threads: %u\n", threads_per_block * blocks);

    // // Record start event
    // check_error(cudaEventRecord(start_neighborhood, 0), "cuda event record - start neighborhood");

    // _gpu_density_field_relevant_neighborhood<<<blocks, threads_per_block>>>(
    //     density_field->field,
    //     d_cells,
    //     d_occupations,
    //     c_n_atoms
    // );

    // // Record stop event
    // check_error(cudaEventRecord(stop_neighborhood, 0), "cuda event record - stop neighborhood");
    // check_error(cudaEventSynchronize(stop_neighborhood), "cuda event synchronize - stop neighborhood");

    // // Calculate elapsed time
    // check_error(cudaEventElapsedTime(&elapsedTime_neighborhood, start_neighborhood, stop_neighborhood), "cuda event elapsed time - neighborhood");
    // printf("Neighborhood iteration kernel execution time: %f ms\n", elapsedTime_neighborhood);

    // // Destroy events
    // check_error(cudaEventDestroy(start_neighborhood), "cuda event destroy - start neighborhood");
    // check_error(cudaEventDestroy(stop_neighborhood), "cuda event destroy - stop neighborhood");

    // check_error(cudaDeviceSynchronize(), "neighborhood iteration - synch");
    // check_error(cudaGetLastError(), "neighborhood iteration - get last error");


    // printf("Number of atoms in the molecule: %lu\n", molecule->n_atoms);

    float total_time = elapsedTime + elapsedTime_neighborhood;
    printf("FINAL KERNEL TIME: %f ms\n", total_time);

    // Free allocated memory
    check_error(cudaFree(d_cells), "cuda free - cell list");
    check_error(cudaFree(d_occupations), "cuda free - occupations");
}


void ONE_THREAD_PER_POINT_eddi_gpu_compute_density_field_cell_list_previous_cutoff(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info)
{
    const float c_x = info->cx;
    const float c_y = info->cy;
    const float c_z = info->cz;

    const size_t nc_x = (density_field->x_size * density_field->dx) / c_x;
    const size_t nc_y = (density_field->y_size * density_field->dy) / c_y;
    const size_t nc_z = (density_field->z_size * density_field->dz) / c_z;


    printf("==== DENSITY COMPUTATION (CUTOFF) ====\n");
    printf("Domain size: x_size = %lu, y_size = %lu, z_size = %lu\n", density_field->x_size, density_field->y_size, density_field->z_size);

    const size_t n_points = density_field->x_size * density_field->y_size * density_field->z_size;


    const size_t nc_tot = nc_x * nc_y * nc_z;

    const float multiplier = molecule->n_atoms * 0.5;
    const eddi_size_t c_n_atoms = multiplier + 1;

    size_t threads_per_block = 512;
    size_t blocks = (nc_tot + threads_per_block - 1) / threads_per_block;

    float4* d_cells;
    size_t* d_included_neighbors;
    size_t* d_occupations;
    unsigned int* d_n_included_neighbors;

    _load_cell_list_data_in_constant_memory(nc_x, nc_y, nc_z, c_x, c_y, c_z);

    printf("Cell dimensions: cx = %f, cy = %f, cz = %f\n", c_x, c_y, c_z);
    printf("Number of cells along x-axis (nc_x): %lu\n", nc_x);
    printf("Number of cells along y-axis (nc_y): %lu\n", nc_y);
    printf("Number of cells along z-axis (nc_z): %lu\n", nc_z);
    printf("Total number of cells (nc_tot): %lu\n", nc_tot);
    printf("Number of atoms per cell (c_n_atoms): %lu\n\n\n", c_n_atoms);

    printf("KERNEL: ATOM PARTITIONING\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);

    check_error(cudaMalloc((void**) &d_cells, sizeof(float4) * c_n_atoms * nc_tot), "cuda malloc - cell list");
    check_error(cudaMalloc((void**) &d_occupations, sizeof(size_t) * nc_tot), "cuda malloc - occupations");
    check_error(cudaMalloc((void**) &d_included_neighbors, sizeof(size_t) * nc_tot), "cuda malloc - occupations");
    check_error(cudaMalloc((void**) &d_n_included_neighbors, sizeof(unsigned int)), "cuda malloc - occupations");

    check_error(cudaMemset(d_occupations, 0, sizeof(size_t) * nc_tot), "cuda memset - occupations");
    check_error(cudaMemset(d_n_included_neighbors, 0, sizeof(unsigned int)), "cuda memset - occupations");


    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events
    check_error(cudaEventCreate(&start), "cuda event create - start");
    check_error(cudaEventCreate(&stop), "cuda event create - stop");

    // Record start event
    check_error(cudaEventRecord(start, 0), "cuda event record - start");

    _gpu_density_partition_atoms<<<blocks, threads_per_block>>>(
        d_cells,
        d_occupations,
        c_n_atoms,
        molecule->atoms,
        molecule->n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop, 0), "cuda event record - stop");
    check_error(cudaEventSynchronize(stop), "cuda event synchronize - stop");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime, start, stop), "cuda event elapsed time");
    printf("Atom Partitioning execution time: %f ms\n\n\n", elapsedTime);

    // Destroy events
    check_error(cudaEventDestroy(start), "cuda event destroy - start");
    check_error(cudaEventDestroy(stop), "cuda event destroy - stop");

    check_error(cudaDeviceSynchronize(), "cuda device synchronize - partition atoms");
    check_error(cudaGetLastError(), "cuda get last error - partition atoms");

    cudaEvent_t start_exclude, stop_exclude;
    float elapsedTime_exclude;

    // Create events
    check_error(cudaEventCreate(&start_exclude), "cuda event create - start exclude");
    check_error(cudaEventCreate(&stop_exclude), "cuda event create - stop exclude");

    // Record start event
    check_error(cudaEventRecord(start_exclude, 0), "cuda event record - start exclude");

    printf("KERNEL: EMPTY NEIGHBORHOOD EXCLUSION\n");
    _gpu_exclude_empty_neighborhoods<<<blocks, threads_per_block>>>(
        d_included_neighbors,
        d_n_included_neighbors,
        d_occupations,
        c_n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop_exclude, 0), "cuda event record - stop exclude");
    check_error(cudaEventSynchronize(stop_exclude), "cuda event synchronize - stop exclude");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime_exclude, start_exclude, stop_exclude), "cuda event elapsed time - exclude");
    check_error(cudaDeviceSynchronize(), "cuda device synchronize - exclude empty neighborhoods");

    unsigned int n_included_neighbors_host;
    check_error(cudaMemcpy(&n_included_neighbors_host, d_n_included_neighbors, sizeof(unsigned int), cudaMemcpyDeviceToHost), "cuda memcpy - n_included_neighbors");

    size_t total_neighborhoods = nc_tot;
    size_t empty_neighborhoods = total_neighborhoods - n_included_neighbors_host;
    
    float empty_percentage = (float)empty_neighborhoods / total_neighborhoods * 100.0f;
    
    printf("Total neighborhoods: %lu\n", total_neighborhoods);
    printf("Empty neighborhoods: %lu\n", empty_neighborhoods);
    printf("Percentage of empty neighborhoods: %.2f%%\n", empty_percentage);
    printf("Exclude Empty Neighborhoods execution time: %f ms\n\n\n", elapsedTime_exclude);

    // Destroy events
    check_error(cudaEventDestroy(start_exclude), "cuda event destroy - start exclude");
    check_error(cudaEventDestroy(stop_exclude), "cuda event destroy - stop exclude");

    
    printf("KERNEL: DENSITY COMPUTATION\n");
    const size_t points_per_bin_x = (info->cx / density_field->dx);
    const size_t points_per_bin_y = (info->cy / density_field->dy);
    const size_t points_per_bin_z = (info->cz / density_field->dz);
    const size_t points_per_bin = points_per_bin_x * points_per_bin_y * points_per_bin_z;

    printf("Points per bin along x: %lu\n", points_per_bin_x);
    printf("Points per bin along y: %lu\n", points_per_bin_y);
    printf("Points per bin along z: %lu\n", points_per_bin_z);
    printf("Total Points per bin  : %lu\n", points_per_bin);


    threads_per_block = 256;
    blocks = (n_included_neighbors_host * points_per_bin + threads_per_block - 1) / threads_per_block;

    printf("Excluded Neighborhood Kernel Configuration:\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);



    cudaEvent_t start_neighborhood, stop_neighborhood;
    float elapsedTime_neighborhood;

    // Create events
    check_error(cudaEventCreate(&start_neighborhood), "cuda event create - start neighborhood");
    check_error(cudaEventCreate(&stop_neighborhood), "cuda event create - stop neighborhood");

    // Record start event
    check_error(cudaEventRecord(start_neighborhood, 0), "cuda event record - start neighborhood");

    _gpu_density_field_excluded_neighborhood<<<blocks, threads_per_block>>>(
        density_field->field,
        d_cells,
        d_included_neighbors,
        points_per_bin_x,
        points_per_bin_y,
        points_per_bin_z,
        d_n_included_neighbors,
        c_n_atoms,
        d_occupations
    );

    // Record stop event
    check_error(cudaEventRecord(stop_neighborhood, 0), "cuda event record - stop neighborhood");
    check_error(cudaEventSynchronize(stop_neighborhood), "cuda event synchronize - stop neighborhood");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime_neighborhood, start_neighborhood, stop_neighborhood), "cuda event elapsed time - neighborhood");
    printf("Excluded Neighborhood Kernel execution time: %f ms\n\n\n", elapsedTime_neighborhood);

    // Destroy events
    check_error(cudaEventDestroy(start_neighborhood), "cuda event destroy - start neighborhood");
    check_error(cudaEventDestroy(stop_neighborhood), "cuda event destroy - stop neighborhood");

    check_error(cudaDeviceSynchronize(), "cuda device synchronize - density computation");

    float total_time = elapsedTime + elapsedTime_exclude + elapsedTime_neighborhood;
    printf("FINAL KERNEL TIME: %f ms\n", total_time);
    // Free allocated memory
    check_error(cudaFree(d_cells), "cuda free - cell list");
    check_error(cudaFree(d_occupations), "cuda free - occupations");
    check_error(cudaFree(d_included_neighbors), "cuda free - included neighbors");
    check_error(cudaFree(d_n_included_neighbors), "cuda free - number of included neighbors");

}


void ONE_BLOCK_PER_BIN_eddi_gpu_compute_density_field_cell_list_previous_cutoff(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info)
{
    const float c_x = info->cx;
    const float c_y = info->cy;
    const float c_z = info->cz;

    const size_t nc_x = (density_field->x_size * density_field->dx) / c_x;
    const size_t nc_y = (density_field->y_size * density_field->dy) / c_y;
    const size_t nc_z = (density_field->z_size * density_field->dz) / c_z;


    printf("==== DENSITY COMPUTATION (ONE BLOCK PER BIN) ====\n");

    printf("Domain size: x_size = %lu, y_size = %lu, z_size = %lu\n", density_field->x_size, density_field->y_size, density_field->z_size);

    const size_t n_points = density_field->x_size * density_field->y_size * density_field->z_size;


    const size_t nc_tot = nc_x * nc_y * nc_z;

    const float multiplier = molecule->n_atoms * 0.5;
    const eddi_size_t c_n_atoms = 1024;

    size_t threads_per_block = 512;
    // TODO: choose value of blocks
    size_t blocks = (molecule->n_atoms + threads_per_block - 1) / threads_per_block;

    float4* d_cells;
    size_t* d_included_neighbors;
    size_t* d_occupations;
    unsigned int* d_n_included_neighbors;

    _load_cell_list_data_in_constant_memory(nc_x, nc_y, nc_z, c_x, c_y, c_z);

    printf("Cell dimensions: cx = %f, cy = %f, cz = %f\n", c_x, c_y, c_z);
    printf("Number of cells along x-axis (nc_x): %lu\n", nc_x);
    printf("Number of cells along y-axis (nc_y): %lu\n", nc_y);
    printf("Number of cells along z-axis (nc_z): %lu\n", nc_z);
    printf("Total number of cells (nc_tot): %lu\n", nc_tot);
    printf("Number of atoms per cell (c_n_atoms): %lu\n\n\n", c_n_atoms);

    printf("KERNEL: ATOM PARTITIONING\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);

    check_error(cudaMalloc((void**) &d_cells, sizeof(float4) * c_n_atoms * nc_tot), "cuda malloc - cell list");
    check_error(cudaMalloc((void**) &d_occupations, sizeof(size_t) * nc_tot), "cuda malloc - occupations");
    check_error(cudaMalloc((void**) &d_included_neighbors, sizeof(size_t) * nc_tot), "cuda malloc - occupations");
    check_error(cudaMalloc((void**) &d_n_included_neighbors, sizeof(unsigned int)), "cuda malloc - occupations");

    check_error(cudaMemset(d_occupations, 0, sizeof(size_t) * nc_tot), "cuda memset - occupations");
    check_error(cudaMemset(d_n_included_neighbors, 0, sizeof(unsigned int)), "cuda memset - occupations");


    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events
    check_error(cudaEventCreate(&start), "cuda event create - start");
    check_error(cudaEventCreate(&stop), "cuda event create - stop");

    // Record start event
    check_error(cudaEventRecord(start, 0), "cuda event record - start");

    _gpu_density_partition_atoms_ATOM_BASED<<<blocks, threads_per_block>>>(
        d_cells,
        d_occupations,
        c_n_atoms,
        molecule->atoms,
        molecule->n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop, 0), "cuda event record - stop");
    check_error(cudaEventSynchronize(stop), "cuda event synchronize - stop");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime, start, stop), "cuda event elapsed time");
    printf("Atom Partitioning execution time: %f ms\n\n\n", elapsedTime);

    // Destroy events
    check_error(cudaEventDestroy(start), "cuda event destroy - start");
    check_error(cudaEventDestroy(stop), "cuda event destroy - stop");

    check_error(cudaDeviceSynchronize(), "cuda device synchronize - partition atoms");
    check_error(cudaGetLastError(), "cuda get last error - partition atoms");

    cudaEvent_t start_exclude, stop_exclude;
    float elapsedTime_exclude;

    // Create events
    check_error(cudaEventCreate(&start_exclude), "cuda event create - start exclude");
    check_error(cudaEventCreate(&stop_exclude), "cuda event create - stop exclude");

    // Record start event
    check_error(cudaEventRecord(start_exclude, 0), "cuda event record - start exclude");

    printf("KERNEL: EMPTY NEIGHBORHOOD EXCLUSION\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);
    _gpu_exclude_empty_neighborhoods<<<blocks, threads_per_block>>>(
        d_included_neighbors,
        d_n_included_neighbors,
        d_occupations,
        c_n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop_exclude, 0), "cuda event record - stop exclude");
    check_error(cudaEventSynchronize(stop_exclude), "cuda event synchronize - stop exclude");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime_exclude, start_exclude, stop_exclude), "cuda event elapsed time - exclude");
    check_error(cudaDeviceSynchronize(), "cuda device synchronize - exclude empty neighborhoods");

    unsigned int n_included_neighbors_host;
    check_error(cudaMemcpy(&n_included_neighbors_host, d_n_included_neighbors, sizeof(unsigned int), cudaMemcpyDeviceToHost), "cuda memcpy - n_included_neighbors");

    size_t total_neighborhoods = nc_tot;
    size_t empty_neighborhoods = total_neighborhoods - n_included_neighbors_host;
    
    float empty_percentage = (float)empty_neighborhoods / total_neighborhoods * 100.0f;
    
    printf("Total neighborhoods: %lu\n", total_neighborhoods);
    printf("Empty neighborhoods: %lu\n", empty_neighborhoods);
    printf("Percentage of empty neighborhoods: %.2f%%\n", empty_percentage);
    printf("Exclude Empty Neighborhoods execution time: %f ms\n\n\n", elapsedTime_exclude);

    // Destroy events
    check_error(cudaEventDestroy(start_exclude), "cuda event destroy - start exclude");
    check_error(cudaEventDestroy(stop_exclude), "cuda event destroy - stop exclude");

    
    printf("KERNEL: DENSITY COMPUTATION\n");
    const size_t points_per_bin_x = (info->cx / density_field->dx);
    const size_t points_per_bin_y = (info->cy / density_field->dy);
    const size_t points_per_bin_z = (info->cz / density_field->dz);
    const size_t points_per_bin = points_per_bin_x * points_per_bin_y * points_per_bin_z;

    printf("Points per bin along x: %lu\n", points_per_bin_x);
    printf("Points per bin along y: %lu\n", points_per_bin_y);
    printf("Points per bin along z: %lu\n", points_per_bin_z);
    printf("Total Points per bin  : %lu\n", points_per_bin);


    threads_per_block = 512;
    blocks = n_included_neighbors_host;

    printf("Excluded Neighborhood Kernel Configuration:\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);

    cudaEvent_t start_neighborhood, stop_neighborhood;
    float elapsedTime_neighborhood;

    // Create events
    check_error(cudaEventCreate(&start_neighborhood), "cuda event create - start neighborhood");
    check_error(cudaEventCreate(&stop_neighborhood), "cuda event create - stop neighborhood");

    // Record start event
    check_error(cudaEventRecord(start_neighborhood, 0), "cuda event record - start neighborhood");

    size_t shared_mem = 2048 * sizeof(float4);

    _gpu_density_field_excluded_neighborhood_ONE_BLOCK_PER_BIN<<<blocks, threads_per_block, shared_mem>>>(
        density_field->field,
        d_cells,
        d_included_neighbors,
        points_per_bin_x,
        points_per_bin_y,
        points_per_bin_z,
        d_n_included_neighbors,
        c_n_atoms,
        d_occupations
    );
    check_error(cudaDeviceSynchronize(), "cuda device synchronize - excluded neighborhood kernel");
    check_error(cudaGetLastError(), "cuda get last error - excluded neighborhood kernel");

    // Record stop event
    check_error(cudaEventRecord(stop_neighborhood, 0), "cuda event record - stop neighborhood");
    check_error(cudaEventSynchronize(stop_neighborhood), "cuda event synchronize - stop neighborhood");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime_neighborhood, start_neighborhood, stop_neighborhood), "cuda event elapsed time - neighborhood");
    printf("Excluded Neighborhood Kernel execution time: %f ms\n\n\n", elapsedTime_neighborhood);

    // Destroy events
    check_error(cudaEventDestroy(start_neighborhood), "cuda event destroy - start neighborhood");
    check_error(cudaEventDestroy(stop_neighborhood), "cuda event destroy - stop neighborhood");


    float total_time = elapsedTime + elapsedTime_exclude + elapsedTime_neighborhood;
    printf("FINAL KERNEL TIME: %f ms\n", total_time);

    // Free allocated memory
    check_error(cudaFree(d_cells), "cuda free - cell list");
    check_error(cudaFree(d_occupations), "cuda free - occupations");
    check_error(cudaFree(d_included_neighbors), "cuda free - included neighbors");
    check_error(cudaFree(d_n_included_neighbors), "cuda free - number of included neighbors");

}

void OPT_ONE_BLOCK_PER_BIN_eddi_gpu_compute_density_field_cell_list_previous_cutoff(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info)
{
    const float c_x = info->cx;
    const float c_y = info->cy;
    const float c_z = info->cz;

    const size_t nc_x = (density_field->x_size * density_field->dx) / c_x;
    const size_t nc_y = (density_field->y_size * density_field->dy) / c_y;
    const size_t nc_z = (density_field->z_size * density_field->dz) / c_z;


    printf("==== DENSITY COMPUTATION (ONE BLOCK PER BIN - OPTIMIZED ACCESS) ====\n");
    printf("Domain size: x_size = %lu, y_size = %lu, z_size = %lu\n", density_field->x_size, density_field->y_size, density_field->z_size);

    const size_t n_points = density_field->x_size * density_field->y_size * density_field->z_size;


    const size_t nc_tot = nc_x * nc_y * nc_z;

    const float multiplier = molecule->n_atoms * 0.5;
    const eddi_size_t c_n_atoms = 1024;

    size_t threads_per_block = 512;
    // TODO: choose value of blocks
    size_t blocks = (molecule->n_atoms + threads_per_block - 1) / threads_per_block;

    float4* d_cells;
    size_t* d_included_neighbors;
    size_t* d_occupations;
    unsigned int* d_n_included_neighbors;

    _load_cell_list_data_in_constant_memory(nc_x, nc_y, nc_z, c_x, c_y, c_z);

    printf("Cell dimensions: cx = %f, cy = %f, cz = %f\n", c_x, c_y, c_z);
    printf("Number of cells along x-axis (nc_x): %lu\n", nc_x);
    printf("Number of cells along y-axis (nc_y): %lu\n", nc_y);
    printf("Number of cells along z-axis (nc_z): %lu\n", nc_z);
    printf("Total number of cells (nc_tot): %lu\n", nc_tot);
    printf("Number of atoms per cell (c_n_atoms): %lu\n\n\n", c_n_atoms);

    printf("KERNEL: ATOM PARTITIONING\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);

    check_error(cudaMalloc((void**) &d_cells, sizeof(float4) * c_n_atoms * nc_tot), "cuda malloc - cell list");
    check_error(cudaMalloc((void**) &d_occupations, sizeof(size_t) * nc_tot), "cuda malloc - occupations");
    check_error(cudaMalloc((void**) &d_included_neighbors, sizeof(size_t) * nc_tot), "cuda malloc - occupations");
    check_error(cudaMalloc((void**) &d_n_included_neighbors, sizeof(unsigned int)), "cuda malloc - occupations");

    check_error(cudaMemset(d_occupations, 0, sizeof(size_t) * nc_tot), "cuda memset - occupations");
    check_error(cudaMemset(d_n_included_neighbors, 0, sizeof(unsigned int)), "cuda memset - occupations");


    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events
    check_error(cudaEventCreate(&start), "cuda event create - start");
    check_error(cudaEventCreate(&stop), "cuda event create - stop");

    // Record start event
    check_error(cudaEventRecord(start, 0), "cuda event record - start");

    _gpu_density_partition_atoms_ATOM_BASED<<<blocks, threads_per_block>>>(
        d_cells,
        d_occupations,
        c_n_atoms,
        molecule->atoms,
        molecule->n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop, 0), "cuda event record - stop");
    check_error(cudaEventSynchronize(stop), "cuda event synchronize - stop");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime, start, stop), "cuda event elapsed time");
    printf("Atom Partitioning execution time: %f ms\n\n\n", elapsedTime);

    // Destroy events
    check_error(cudaEventDestroy(start), "cuda event destroy - start");
    check_error(cudaEventDestroy(stop), "cuda event destroy - stop");

    check_error(cudaDeviceSynchronize(), "cuda device synchronize - partition atoms");
    check_error(cudaGetLastError(), "cuda get last error - partition atoms");

    cudaEvent_t start_exclude, stop_exclude;
    float elapsedTime_exclude;

    // Create events
    check_error(cudaEventCreate(&start_exclude), "cuda event create - start exclude");
    check_error(cudaEventCreate(&stop_exclude), "cuda event create - stop exclude");

    // Record start event
    check_error(cudaEventRecord(start_exclude, 0), "cuda event record - start exclude");

    printf("KERNEL: EMPTY NEIGHBORHOOD EXCLUSION\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);
    _gpu_exclude_empty_neighborhoods<<<blocks, threads_per_block>>>(
        d_included_neighbors,
        d_n_included_neighbors,
        d_occupations,
        c_n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop_exclude, 0), "cuda event record - stop exclude");
    check_error(cudaEventSynchronize(stop_exclude), "cuda event synchronize - stop exclude");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime_exclude, start_exclude, stop_exclude), "cuda event elapsed time - exclude");
    check_error(cudaDeviceSynchronize(), "cuda device synchronize - exclude empty neighborhoods");

    unsigned int n_included_neighbors_host;
    check_error(cudaMemcpy(&n_included_neighbors_host, d_n_included_neighbors, sizeof(unsigned int), cudaMemcpyDeviceToHost), "cuda memcpy - n_included_neighbors");

    size_t total_neighborhoods = nc_tot;
    size_t empty_neighborhoods = total_neighborhoods - n_included_neighbors_host;
    
    float empty_percentage = (float)empty_neighborhoods / total_neighborhoods * 100.0f;
    
    printf("Total neighborhoods: %lu\n", total_neighborhoods);
    printf("Empty neighborhoods: %lu\n", empty_neighborhoods);
    printf("Percentage of empty neighborhoods: %.2f%%\n", empty_percentage);
    printf("Exclude Empty Neighborhoods execution time: %f ms\n\n\n", elapsedTime_exclude);

    // Destroy events
    check_error(cudaEventDestroy(start_exclude), "cuda event destroy - start exclude");
    check_error(cudaEventDestroy(stop_exclude), "cuda event destroy - stop exclude");

    
    printf("KERNEL: DENSITY COMPUTATION\n");
    const size_t points_per_bin_x = (info->cx / density_field->dx);
    const size_t points_per_bin_y = (info->cy / density_field->dy);
    const size_t points_per_bin_z = (info->cz / density_field->dz);
    const size_t points_per_bin = points_per_bin_x * points_per_bin_y * points_per_bin_z;

    printf("Points per bin along x: %lu\n", points_per_bin_x);
    printf("Points per bin along y: %lu\n", points_per_bin_y);
    printf("Points per bin along z: %lu\n", points_per_bin_z);
    printf("Total Points per bin  : %lu\n", points_per_bin);


    threads_per_block = 512;
    blocks = n_included_neighbors_host;

    printf("Excluded Neighborhood Kernel Configuration:\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);

    cudaEvent_t start_neighborhood, stop_neighborhood;
    float elapsedTime_neighborhood;

    // Create events
    check_error(cudaEventCreate(&start_neighborhood), "cuda event create - start neighborhood");
    check_error(cudaEventCreate(&stop_neighborhood), "cuda event create - stop neighborhood");

    // Record start event
    check_error(cudaEventRecord(start_neighborhood, 0), "cuda event record - start neighborhood");

    size_t shared_mem = 2048 * sizeof(float4) + 27 * sizeof(size_t) * 2;

    _gpu_density_field_excluded_neighborhood_ONE_BLOCK_PER_BIN_OPT<<<blocks, threads_per_block, shared_mem>>>(
        density_field->field,
        d_cells,
        d_included_neighbors,
        points_per_bin_x,
        points_per_bin_y,
        points_per_bin_z,
        d_n_included_neighbors,
        c_n_atoms,
        d_occupations
    );
    check_error(cudaDeviceSynchronize(), "cuda device synchronize - excluded neighborhood kernel");
    check_error(cudaGetLastError(), "cuda get last error - excluded neighborhood kernel");

    // Record stop event
    check_error(cudaEventRecord(stop_neighborhood, 0), "cuda event record - stop neighborhood");
    check_error(cudaEventSynchronize(stop_neighborhood), "cuda event synchronize - stop neighborhood");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime_neighborhood, start_neighborhood, stop_neighborhood), "cuda event elapsed time - neighborhood");
    printf("Excluded Neighborhood Kernel execution time: %f ms\n\n\n", elapsedTime_neighborhood);

    // Destroy events
    check_error(cudaEventDestroy(start_neighborhood), "cuda event destroy - start neighborhood");
    check_error(cudaEventDestroy(stop_neighborhood), "cuda event destroy - stop neighborhood");


    float total_time = elapsedTime + elapsedTime_exclude + elapsedTime_neighborhood;
    printf("FINAL KERNEL TIME: %f ms\n", total_time);

    // Free allocated memory
    check_error(cudaFree(d_cells), "cuda free - cell list");
    check_error(cudaFree(d_occupations), "cuda free - occupations");
    check_error(cudaFree(d_included_neighbors), "cuda free - included neighbors");
    check_error(cudaFree(d_n_included_neighbors), "cuda free - number of included neighbors");

}


void ONE_BLOCK_PER_POINT_eddi_gpu_compute_density_field_cell_list_previous_cutoff(eddi_gpu_density_field_t* density_field, eddi_gpu_molecule_t* molecule, eddi_cl_info_t* info)
{
    const float c_x = info->cx;
    const float c_y = info->cy;
    const float c_z = info->cz;

    const size_t nc_x = (density_field->x_size * density_field->dx) / c_x;
    const size_t nc_y = (density_field->y_size * density_field->dy) / c_y;
    const size_t nc_z = (density_field->z_size * density_field->dz) / c_z;


    printf("==== DENSITY COMPUTATION (CUTOFF - ONE BLOCK PER POINT) ====\n");
    printf("Domain size: x_size = %lu, y_size = %lu, z_size = %lu\n", density_field->x_size, density_field->y_size, density_field->z_size);

    const size_t n_points = density_field->x_size * density_field->y_size * density_field->z_size;


    const size_t nc_tot = nc_x * nc_y * nc_z;

    const float multiplier = molecule->n_atoms * 0.5;
    const eddi_size_t c_n_atoms = 1024;

    size_t threads_per_block = 32;
    size_t blocks = (nc_tot + threads_per_block - 1) / threads_per_block;

    float4* d_cells;
    size_t* d_included_neighbors;
    size_t* d_occupations;
    unsigned int* d_n_included_neighbors;

    _load_cell_list_data_in_constant_memory(nc_x, nc_y, nc_z, c_x, c_y, c_z);

    printf("Cell dimensions: cx = %f, cy = %f, cz = %f\n", c_x, c_y, c_z);
    printf("Number of cells along x-axis (nc_x): %lu\n", nc_x);
    printf("Number of cells along y-axis (nc_y): %lu\n", nc_y);
    printf("Number of cells along z-axis (nc_z): %lu\n", nc_z);
    printf("Total number of cells (nc_tot): %lu\n", nc_tot);
    printf("Number of atoms per cell (c_n_atoms): %lu\n\n\n", c_n_atoms);

    printf("KERNEL: ATOM PARTITIONING\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Total threads: %lu\n", threads_per_block * blocks);

    check_error(cudaMalloc((void**) &d_cells, sizeof(float4) * c_n_atoms * nc_tot), "cuda malloc - cell list");
    check_error(cudaMalloc((void**) &d_occupations, sizeof(size_t) * nc_tot), "cuda malloc - occupations");
    check_error(cudaMalloc((void**) &d_included_neighbors, sizeof(size_t) * nc_tot), "cuda malloc - occupations");
    check_error(cudaMalloc((void**) &d_n_included_neighbors, sizeof(unsigned int)), "cuda malloc - occupations");

    check_error(cudaMemset(d_occupations, 0, sizeof(size_t) * nc_tot), "cuda memset - occupations");
    check_error(cudaMemset(d_n_included_neighbors, 0, sizeof(unsigned int)), "cuda memset - occupations");


    cudaEvent_t start, stop;
    float elapsedTime;

    // Create events
    check_error(cudaEventCreate(&start), "cuda event create - start");
    check_error(cudaEventCreate(&stop), "cuda event create - stop");

    // Record start event
    check_error(cudaEventRecord(start, 0), "cuda event record - start");

    _gpu_density_partition_atoms<<<blocks, threads_per_block>>>(
        d_cells,
        d_occupations,
        c_n_atoms,
        molecule->atoms,
        molecule->n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop, 0), "cuda event record - stop");
    check_error(cudaEventSynchronize(stop), "cuda event synchronize - stop");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime, start, stop), "cuda event elapsed time");
    printf("Atom Partitioning execution time: %f ms\n\n\n", elapsedTime);

    // Destroy events
    check_error(cudaEventDestroy(start), "cuda event destroy - start");
    check_error(cudaEventDestroy(stop), "cuda event destroy - stop");

    check_error(cudaDeviceSynchronize(), "cuda device synchronize - partition atoms");
    check_error(cudaGetLastError(), "cuda get last error - partition atoms");

    cudaEvent_t start_exclude, stop_exclude;
    float elapsedTime_exclude;

    // Create events
    check_error(cudaEventCreate(&start_exclude), "cuda event create - start exclude");
    check_error(cudaEventCreate(&stop_exclude), "cuda event create - stop exclude");

    // Record start event
    check_error(cudaEventRecord(start_exclude, 0), "cuda event record - start exclude");

    printf("KERNEL: EMPTY NEIGHBORHOOD EXCLUSION\n");
    _gpu_exclude_empty_neighborhoods<<<blocks, threads_per_block>>>(
        d_included_neighbors,
        d_n_included_neighbors,
        d_occupations,
        c_n_atoms
    );

    // Record stop event
    check_error(cudaEventRecord(stop_exclude, 0), "cuda event record - stop exclude");
    check_error(cudaEventSynchronize(stop_exclude), "cuda event synchronize - stop exclude");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime_exclude, start_exclude, stop_exclude), "cuda event elapsed time - exclude");
    check_error(cudaDeviceSynchronize(), "cuda device synchronize - exclude empty neighborhoods");

    unsigned int n_included_neighbors_host;
    check_error(cudaMemcpy(&n_included_neighbors_host, d_n_included_neighbors, sizeof(unsigned int), cudaMemcpyDeviceToHost), "cuda memcpy - n_included_neighbors");

    size_t total_neighborhoods = nc_tot;
    size_t empty_neighborhoods = total_neighborhoods - n_included_neighbors_host;
    
    float empty_percentage = (float)empty_neighborhoods / total_neighborhoods * 100.0f;
    
    printf("Total neighborhoods: %lu\n", total_neighborhoods);
    printf("Empty neighborhoods: %lu\n", empty_neighborhoods);
    printf("Percentage of empty neighborhoods: %.2f%%\n", empty_percentage);
    printf("Exclude Empty Neighborhoods execution time: %f ms\n\n\n", elapsedTime_exclude);

    // Destroy events
    check_error(cudaEventDestroy(start_exclude), "cuda event destroy - start exclude");
    check_error(cudaEventDestroy(stop_exclude), "cuda event destroy - stop exclude");

    
    printf("KERNEL: DENSITY COMPUTATION\n");
    const size_t points_per_bin_x = (info->cx / density_field->dx);
    const size_t points_per_bin_y = (info->cy / density_field->dy);
    const size_t points_per_bin_z = (info->cz / density_field->dz);
    const size_t points_per_bin = points_per_bin_x * points_per_bin_y * points_per_bin_z;

    printf("Points per bin along x: %lu\n", points_per_bin_x);
    printf("Points per bin along y: %lu\n", points_per_bin_y);
    printf("Points per bin along z: %lu\n", points_per_bin_z);
    printf("Total Points per bin  : %lu\n", points_per_bin);


    threads_per_block = 512;
    blocks = points_per_bin * (n_included_neighbors_host);
    size_t shared_mem_size = threads_per_block * sizeof(float) + 1024 * sizeof(float4);

    printf("Excluded Neighborhood Kernel Configuration:\n");
    printf("Threads per block: %lu\n", threads_per_block);
    printf("Number of blocks: %lu\n", blocks);
    printf("Shared memory: %lu\n", shared_mem_size);
    printf("Total threads: %lu\n", threads_per_block * blocks);




    cudaEvent_t start_neighborhood, stop_neighborhood;
    float elapsedTime_neighborhood;

    // Create events
    check_error(cudaEventCreate(&start_neighborhood), "cuda event create - start neighborhood");
    check_error(cudaEventCreate(&stop_neighborhood), "cuda event create - stop neighborhood");

    // Record start event
    check_error(cudaEventRecord(start_neighborhood, 0), "cuda event record - start neighborhood");


    _gpu_density_field_excluded_neighborhood_ONE_BLOCK_PER_POINT<<<blocks, threads_per_block, shared_mem_size>>>(
        density_field->field,
        d_cells,
        d_included_neighbors,
        points_per_bin_x,
        points_per_bin_y,
        points_per_bin_z,
        d_n_included_neighbors,
        c_n_atoms,
        d_occupations
    );

    // Record stop event
    check_error(cudaEventRecord(stop_neighborhood, 0), "cuda event record - stop neighborhood");
    check_error(cudaEventSynchronize(stop_neighborhood), "cuda event synchronize - stop neighborhood");

    // Calculate elapsed time
    check_error(cudaEventElapsedTime(&elapsedTime_neighborhood, start_neighborhood, stop_neighborhood), "cuda event elapsed time - neighborhood");
    printf("Excluded Neighborhood Kernel execution time: %f ms\n\n\n", elapsedTime_neighborhood);

    // Destroy events
    check_error(cudaEventDestroy(start_neighborhood), "cuda event destroy - start neighborhood");
    check_error(cudaEventDestroy(stop_neighborhood), "cuda event destroy - stop neighborhood");

    cudaDeviceSynchronize();

    // Free allocated memory
    check_error(cudaFree(d_cells), "cuda free - cell list");
    check_error(cudaFree(d_occupations), "cuda free - occupations");
    check_error(cudaFree(d_included_neighbors), "cuda free - included neighbors");
    check_error(cudaFree(d_n_included_neighbors), "cuda free - number of included neighbors");


}

