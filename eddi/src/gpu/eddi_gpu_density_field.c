#include "eddi_gpu_density_field.h"

bool eddi_gpu_new_density_field(
    eddi_gpu_density_field_t* density_field,
    const eddi_real_t dx,
    const eddi_real_t dy,
    const eddi_real_t dz,
    const eddi_point_t* origin,
    const eddi_size_t n_x,
    const eddi_size_t n_y,
    const eddi_size_t n_z
)
{
    density_field->x_size = n_x;
    density_field->y_size = n_y;
    density_field->z_size = n_z;

    const size_t required_bytes = sizeof(eddi_real_t) * density_field->x_size * density_field->y_size * density_field->z_size;
    printf("Allocating %zu bytes for the density field.\n", required_bytes);
    

    // Allocate the density field on the GPU
    check_error(cudaMalloc((void**) &(density_field->field), sizeof(eddi_real_t) * density_field->x_size * density_field->y_size * density_field->z_size),
                "cuda malloc - field");
    check_error(cudaMemset((void*) density_field->field, 0, sizeof(eddi_real_t) * density_field->x_size * density_field->y_size * density_field->z_size),
                "cuda memset - field");
    
    _load_origin_in_constant_memory(origin);
    _load_field_data_in_constant_memory(
        n_x, n_y, n_z,
        dx, dy, dz
    );

    density_field->dx = dx;
    density_field->dy = dy;
    density_field->dz = dz;
    memcpy(&(density_field->origin), origin, 3 * sizeof(eddi_real_t));

    printf("===== DOMAIN CREATED =====\n");
    printf("Domain Information:\n");
    printf("Origin: (%f, %f, %f)\n", density_field->origin.x, density_field->origin.y, density_field->origin.z);
    printf("Grid Resolution:\n");
    printf("dx: %f, dy: %f, dz: %f\n", density_field->dx, density_field->dy, density_field->dz);
    printf("Grid Dimensions:\n");
    printf("x_size: %zu, y_size: %zu, z_size: %zu\n\n", density_field->x_size, density_field->y_size, density_field->z_size);
    
    return EDDI_RETURN_SUCCESS;

}

bool eddi_gpu_init_field_from_molecule(
    eddi_gpu_density_field_t* density_field,
    eddi_molecule_boundary_t* boundaries,
    const eddi_real_t padding,
    const eddi_real_t dx,
    const eddi_real_t dy,
    const eddi_real_t dz
)
{
    eddi_real_t min_x = boundaries->min_x - padding; 
    eddi_real_t max_x = boundaries->max_x + padding; 
    eddi_real_t min_y = boundaries->min_y - padding; 
    eddi_real_t max_y = boundaries->max_y + padding; 
    eddi_real_t min_z = boundaries->min_z - padding; 
    eddi_real_t max_z = boundaries->max_z + padding;
    
    max_x = min_x + ceil((max_x - min_x) / dx) * dx;
    max_y = min_y + ceil((max_y - min_y) / dy) * dy;
    max_z = min_z + ceil((max_z - min_z) / dz) * dz;

    eddi_size_t n_x = (eddi_size_t)((max_x - min_x) / dx + 0.5);
    eddi_size_t n_y = (eddi_size_t)((max_y - min_y) / dy + 0.5);
    eddi_size_t n_z = (eddi_size_t)((max_z - min_z) / dz + 0.5);


    eddi_point_t origin = {.x = min_x, .y = min_y, .z = min_z};

    return eddi_gpu_new_density_field(
        density_field, 
        dx,
        dy,
        dz,
        &origin,
        n_x,
        n_y,
        n_z
    );
}

bool eddi_gpu_init_field_from_molecule_pts_per_bin(
    eddi_gpu_density_field_t* density_field,
    eddi_molecule_boundary_t* boundaries,
    const eddi_real_t padding,
    const eddi_cl_info_t* bin_info,
    const eddi_size_t total_points_per_bin
)
{
    eddi_real_t min_x = boundaries->min_x - padding; 
    eddi_real_t max_x = boundaries->max_x + padding; 
    eddi_real_t min_y = boundaries->min_y - padding; 
    eddi_real_t max_y = boundaries->max_y + padding; 
    eddi_real_t min_z = boundaries->min_z - padding; 
    eddi_real_t max_z = boundaries->max_z + padding;

    printf("===== DOMAIN CREATION =====\n");


    printf("> Domain dimensions:\n\n");
    printf("x: %.2f->%.2f\ny: %.2f->%.2f\nz: %.2f->%.2f\n", 
        boundaries->min_x, boundaries->max_x, 
        boundaries->min_y, boundaries->max_y, 
        boundaries->min_z, boundaries->max_z);
    printf("Padding: %.2f\n", padding);
    printf("Padded domain dimensions:\n");
    printf("x: %.2f->%.2f\ny: %.2f->%.2f\nz: %.2f->%.2f\n", 
           min_x, max_x, 
           min_y, max_y, 
           min_z, max_z);


    max_x = min_x + ceil((max_x - min_x) / bin_info->cx) * bin_info->cx;
    max_y = min_y + ceil((max_y - min_y) / bin_info->cy) * bin_info->cx;
    max_z = min_z + ceil((max_z - min_z) / bin_info->cz) * bin_info->cx;


    printf("Adjusted domain dimensions:\n");
    printf("x: %.2f->%.2f (%d bins)\ny: %.2f->%.2f (%d bins)\nz: %.2f->%.2f (%d bins)\n\n", 
           min_x, max_x, (int)((max_x - min_x) / bin_info->cx), 
           min_y, max_y, (int)((max_y - min_y) / bin_info->cy), 
           min_z, max_z, (int)((max_z - min_z) / bin_info->cz));

    printf("> Grid resolution\n");
    
    int i = 0;
    int target = (int) total_points_per_bin;

    eddi_real_t pts_x, pts_y, pts_z;
    pts_x = 1.0;
    pts_y = 1.0;
    pts_z = 1.0;

    while(target > 0)
    {
        switch (i)
        {
            case 0:
                pts_x *= 2;
                break;
            case 1:
                pts_y *= 2;
                break;
            case 2:
                pts_z *= 2;
                break;
            default:
                break;
        }
        target /= 2;
        i = (i+1)%3;
    }

    eddi_real_t dx, dy, dz;

    dx = bin_info->cx / pts_x;
    dy = bin_info->cy / pts_y;
    dz = bin_info->cz / pts_z;

    eddi_size_t n_x = (eddi_size_t)((max_x - min_x) / dx );
    eddi_size_t n_y = (eddi_size_t)((max_y - min_y) / dy );
    eddi_size_t n_z = (eddi_size_t)((max_z - min_z) / dz );

    printf("Adjusted grid resolution:\n");
    printf("dx: %.2f, dy: %.2f, dz: %.2f\n", dx, dy, dz);
    printf("Points per bin:\n");
    printf("px: %f, py: %f, pz: %f\n", pts_x, pts_y, pts_z);
    printf("Points in the domain:\n");
    printf("x: %zu, y: %zu, z: %zu\n", n_x, n_y, n_z);
    printf("Total points in the domain: %zu\n", n_x * n_y * n_z);

    eddi_point_t origin = {.x = min_x, .y = min_y, .z = min_z};


    return eddi_gpu_new_density_field(
        density_field, 
        dx,
        dy,
        dz,
        &origin,
        n_x,
        n_y,
        n_z
    );
}

bool eddi_gpu_init_field_from_molecule_binned(
    eddi_gpu_density_field_t* density,
    eddi_molecule_boundary_t* boundaries,
    const eddi_real_t padding,
    const eddi_real_t dx,
    const eddi_real_t dy,
    const eddi_real_t dz,
    const eddi_cl_info_t* bin_info
)
{
    eddi_real_t min_x = boundaries->min_x - padding; 
    eddi_real_t max_x = boundaries->max_x + padding; 
    eddi_real_t min_y = boundaries->min_y - padding; 
    eddi_real_t max_y = boundaries->max_y + padding; 
    eddi_real_t min_z = boundaries->min_z - padding; 
    eddi_real_t max_z = boundaries->max_z + padding;

    printf("===== DOMAIN CREATION =====\n");


    printf("> Domain dimensions:\n\n");
    printf("x: %.2f->%.2f\ny: %.2f->%.2f\nz: %.2f->%.2f\n", 
        boundaries->min_x, boundaries->max_x, 
        boundaries->min_y, boundaries->max_y, 
        boundaries->min_z, boundaries->max_z);
    printf("Padded domain dimensions (%.2f):\n", padding);
    printf("x: %.2f->%.2f\ny: %.2f->%.2f\nz: %.2f->%.2f\n", 
           min_x, max_x, 
           min_y, max_y, 
           min_z, max_z);


    max_x = min_x + ceil((max_x - min_x) / bin_info->cx) * bin_info->cx;
    max_y = min_y + ceil((max_y - min_y) / bin_info->cy) * bin_info->cx;
    max_z = min_z + ceil((max_z - min_z) / bin_info->cz) * bin_info->cx;

    eddi_real_t _dx = bin_info->cx / ceil(bin_info->cx / dx);
    eddi_real_t _dy = bin_info->cy / ceil(bin_info->cy / dy);
    eddi_real_t _dz = bin_info->cz / ceil(bin_info->cz / dz);

    printf("Adjusted domain dimensions:\n");
    printf("x: %.2f->%.2f (%d bins)\ny: %.2f->%.2f (%d bins)\nz: %.2f->%.2f (%d bins)\n\n", 
           min_x, max_x, (int)((max_x - min_x) / bin_info->cx), 
           min_y, max_y, (int)((max_y - min_y) / bin_info->cy), 
           min_z, max_z, (int)((max_z - min_z) / bin_info->cz));


    printf("> Grid resolution (forcing bin size multiple of res):\n\n");
    printf("dx: %.2f -> %.2f\n", dx, _dx);
    printf("dy: %.2f -> %.2f\n", dy, _dy);
    printf("dz: %.2f -> %.2f\n", dz, _dz);

    eddi_size_t n_x = (eddi_size_t)((max_x - min_x) / _dx);
    eddi_size_t n_y = (eddi_size_t)((max_y - min_y) / _dy);
    eddi_size_t n_z = (eddi_size_t)((max_z - min_z) / _dz);


    printf("Grid dimensions:\n");
    printf("x: %zu, y: %zu, z: %zu\n", n_x, n_y, n_z);
    printf("Total points: %zu\n", n_x * n_y * n_z);
    printf("Points per bin:\n");
    printf("x: %f, y: %f, z: %f\n\n", bin_info->cx / _dx, bin_info->cy / _dy, bin_info->cz / _dz);

    eddi_point_t origin = {.x = min_x, .y = min_y, .z = min_z};

    return eddi_gpu_new_density_field(
        density, 
        dx,
        dy,
        dz,
        &origin,
        n_x,
        n_y,
        n_z
    );
}



void eddi_gpu_get_density_field(const eddi_gpu_density_field_t* device_field, eddi_density_field_t* host_field)
{
    host_field->dx = device_field->dx;
    host_field->dy = device_field->dy;
    host_field->dz = device_field->dz;

    host_field->x_size = device_field->x_size;
    host_field->y_size = device_field->y_size;
    host_field->z_size = device_field->z_size;

    const eddi_size_t size = host_field->x_size * host_field->y_size * host_field->z_size;

    host_field->origin.x = device_field->origin.x;
    host_field->origin.y = device_field->origin.y;
    host_field->origin.z = device_field->origin.z;
    host_field->field = (eddi_real_t*) malloc(sizeof(eddi_real_t) * size);
    check_error(cudaMemcpy(host_field->field, device_field->field, sizeof(eddi_real_t) * size, cudaMemcpyDeviceToHost), "cuda memcpy - get field");
}


void eddi_gpu_free_density_field(eddi_gpu_density_field_t* density_field)
{
    check_error(cudaDeviceSynchronize(), "cuda synchronize - free density field");
    if (density_field->field != NULL)
    {
        check_error(cudaFree(density_field->field), "cuda free - density field");
        density_field->field = NULL;
    }
}