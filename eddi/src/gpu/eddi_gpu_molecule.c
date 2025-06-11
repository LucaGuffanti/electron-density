#include "eddi_gpu_molecule.h"

bool eddi_molecule_cpu_to_gpu(const eddi_molecule_t* cpu_molecule, eddi_gpu_molecule_t* gpu_molecule)
{
    gpu_molecule->n_atoms = cpu_molecule->n_atoms;
    gpu_molecule->atoms = (eddi_gpu_atom_t*) malloc(sizeof(eddi_gpu_atom_t) * gpu_molecule->n_atoms);

    if (!gpu_molecule)
    {
        EDDI_DEBUG_PRINT("[ERROR] Could not allocate molecule with GPU-based format (CPU)\n");
        return EDDI_RETURN_FAILURE;
    }

    #pragma omp parallel for
    for (eddi_size_t i = 0; i < cpu_molecule->n_atoms; ++i)
    {
        gpu_molecule->atoms[i].x = cpu_molecule->atoms_x[i];
        gpu_molecule->atoms[i].y = cpu_molecule->atoms_y[i];
        gpu_molecule->atoms[i].z = cpu_molecule->atoms_z[i];
        gpu_molecule->atoms[i].w = cpu_molecule->atomic_numbers[i];

    }

    return EDDI_RETURN_SUCCESS;
}

bool eddi_molecule_gpu_to_cpu(const eddi_gpu_molecule_t* gpu_molecule, eddi_molecule_t* cpu_molecule)
{
    printf("Allocating memory\n");
    printf("%d\n", gpu_molecule->n_atoms);

    cpu_molecule->n_atoms = gpu_molecule->n_atoms;
    cpu_molecule->atoms_x = (eddi_real_t*) malloc(sizeof(eddi_real_t) * cpu_molecule->n_atoms);
    cpu_molecule->atoms_y = (eddi_real_t*) malloc(sizeof(eddi_real_t) * cpu_molecule->n_atoms);
    cpu_molecule->atoms_z = (eddi_real_t*) malloc(sizeof(eddi_real_t) * cpu_molecule->n_atoms);
    cpu_molecule->atomic_numbers = (eddi_atomic_number_t*) malloc(sizeof(eddi_atomic_number_t) * cpu_molecule->n_atoms);

    printf("Allocated memory\n");

    if (!cpu_molecule->atoms_x || !cpu_molecule->atoms_y || !cpu_molecule->atoms_z || !cpu_molecule->atomic_numbers)
    {
        EDDI_DEBUG_PRINT("[ERROR] Could not open file\n");
        return EDDI_TEST_FAILURE;
    }

    #pragma omp parallel for
    for (eddi_size_t i = 0; i < cpu_molecule->n_atoms; ++i)
    {
        // printf("Atom %zu: x=%f, y=%f, z=%f, atomic_number=%d\n", i, gpu_molecule->atoms[i].x, gpu_molecule->atoms[i].y, gpu_molecule->atoms[i].z, (int)gpu_molecule->atoms[i].w);
        cpu_molecule->atoms_x[i] = gpu_molecule->atoms[i].x;
        cpu_molecule->atoms_y[i] = gpu_molecule->atoms[i].y;
        cpu_molecule->atoms_z[i] = gpu_molecule->atoms[i].z;
        cpu_molecule->atomic_numbers[i] = (eddi_atomic_number_t) gpu_molecule->atoms[i].w;
    }
}


bool eddi_gpu_read_pdb(const char* filename, eddi_gpu_molecule_t* device_molecule, eddi_molecule_boundary_t* boundaries)
{
    FILE* fp;

    if (!(fp = fopen(filename, "r")))
    {
        EDDI_DEBUG_PRINT("[ERROR] Could not open file\n");
        return EDDI_TEST_FAILURE;
    }

    EDDI_DEBUG_PRINT("[INFO] Reading pdb file\n");

    eddi_size_t count = 0;

    char type_of_record[10];
    fscanf(fp, "%6c", type_of_record);
    type_of_record[6] = '\0';

    eddi_size_t current_size = 1024;

    eddi_gpu_atom_t* atoms = malloc(sizeof(eddi_gpu_atom_t) * current_size);

    while(!feof(fp))
    {
        // If either an ATOM or HETATM is read than manage it
        if((type_of_record[0] == 'A' && type_of_record[1] == 'T') || (type_of_record[0] == 'H' && type_of_record[3] == 'A'))
        {
            eddi_real_t x;
            eddi_real_t y;
            eddi_real_t z;
            char name[3];
            
            // atom serial number - 5 int
            // atom name - 4 char
            // alternate location indicator - 1 char
            // residue name - 4 char
            // chain identifies - 1 char 
            // residue sequence number - 4 int
            // code for insertion of residues - 1 char
            // x - 8.3 real
            // y - 8.3 real
            // z - 8.3 real
            // occupancy  - 6.2 real
            // temperature factor - 6.2 real
            // segment identifier - 4 char
            
            char x_s[9];
            char y_s[9];
            char z_s[9];
            fscanf(fp, "%*5d%*c%*4c%*c%*3c%*c%*c%*4c%*c%*c%*c%*c%8c%8c%8c%*6c%*6c%*10c%2s", x_s, y_s, z_s, name);

            sscanf(x_s, "%f", &x);
            sscanf(y_s, "%f", &y);
            sscanf(z_s, "%f", &z);

            EDDI_DEBUG_PRINT("%f ", x);
            EDDI_DEBUG_PRINT("%f ", y);
            EDDI_DEBUG_PRINT("%f ", z);
            
            EDDI_DEBUG_PRINT("%s\n", name);

            // PDB files provide atom information in Angstroms!
            x = x * EDDI_ANGSTROM_TO_BOHR;
            y = y * EDDI_ANGSTROM_TO_BOHR;
            z = z * EDDI_ANGSTROM_TO_BOHR;

            // Recover the atomic number from the symbol
            eddi_atomic_number_t atomic_number = eddi_symbol_to_number(name);
            
            if(count == current_size)
            {
                current_size = current_size * 2;
                atoms = realloc(atoms, sizeof(eddi_gpu_atom_t) * current_size);
                if (!atoms)
                {
                    EDDI_DEBUG_PRINT("[ERROR] Could not reallocate memory for atoms\n");
                    fclose(fp);
                    return EDDI_TEST_FAILURE;
                }
                printf("Reallocated. Size %d\n", current_size);
            }

            if (count == 0)
            {
                boundaries->min_x = x;
                boundaries->max_x = x;
                boundaries->min_y = y;
                boundaries->max_y = y;
                boundaries->min_z = z;
                boundaries->max_z = z;
            }
            else
            {
                if (x < boundaries->min_x) boundaries->min_x = x;
                else if (x > boundaries->max_x) boundaries->max_x = x;
                if (y < boundaries->min_y) boundaries->min_y = y;
                else if (y > boundaries->max_y) boundaries->max_y = y;
                if (z < boundaries->min_z) boundaries->min_z = z;
                else if (z > boundaries->max_z) boundaries->max_z = z;
            }

            atoms[count].x = x;
            atoms[count].y = y;
            atoms[count].z = z;
            atoms[count].w = atomic_number;

            count++;
            EDDI_DEBUG_PRINT("%d\n", count);
        }

        fscanf(fp, "%*[^\n]%*c");
        fscanf(fp, "%6c", type_of_record);
    }

    fclose(fp);

    printf("Offloading\n");
    eddi_gpu_molecule_t molecule = {.atoms = atoms, .n_atoms = count};
    device_molecule->atoms = eddi_gpu_offload_molecule(&molecule);
    device_molecule->n_atoms = count;

    printf("%d atoms\n", count);
    return count;
}


bool eddi_gpu_read_coord(const char* filename, eddi_gpu_molecule_t* device_molecule, eddi_molecule_boundary_t* boundaries)
{
    FILE* fp;

    if (!(fp = fopen(filename, "r")))
    {
        EDDI_DEBUG_PRINT("[ERROR] Could not open file\n");
        return EDDI_TEST_FAILURE;
    }

    printf("Reading file: %s\n", filename);

    EDDI_DEBUG_PRINT("[INFO] Reading coord file\n");

    eddi_size_t count = 0;


    eddi_size_t n_atoms;
    fscanf(fp, "%zu", &n_atoms);
    printf("Number of atoms: %zu\n", n_atoms);
    eddi_gpu_atom_t* atoms = malloc(sizeof(eddi_gpu_atom_t) * n_atoms);

    while(!feof(fp))
    {
        // If either an ATOM or HETATM is read than manage it
        eddi_real_t x;
        eddi_real_t y;
        eddi_real_t z;
        int atomic_number;           

        fscanf(fp, " %f %f %f %d ", &x, &y, &z, &atomic_number);

        if ((count+1)%10000 == 0)
            printf("atom %zu/%zu\r", count+1, n_atoms);
        
        if (count == 0)
        {
            boundaries->min_x = x;
            boundaries->max_x = x;
            boundaries->min_y = y;
            boundaries->max_y = y;
            boundaries->min_z = z;
            boundaries->max_z = z;
        }
        else
        {
            if (x < boundaries->min_x) boundaries->min_x = x;
            else if (x > boundaries->max_x) boundaries->max_x = x;
            if (y < boundaries->min_y) boundaries->min_y = y;
            else if (y > boundaries->max_y) boundaries->max_y = y;
            if (z < boundaries->min_z) boundaries->min_z = z;
            else if (z > boundaries->max_z) boundaries->max_z = z;
        }

        atoms[count].x = x;
        atoms[count].y = y;
        atoms[count].z = z;
        atoms[count].w = atomic_number;

        count++;
        EDDI_DEBUG_PRINT("%d\n", count);
    }
    printf("\n");

    
    fclose(fp);

    printf("Offloading\n");
    eddi_gpu_molecule_t molecule = {.atoms = atoms, .n_atoms = count};
    device_molecule->atoms = eddi_gpu_offload_molecule(&molecule);
    device_molecule->n_atoms = count;

    printf("%d atoms\n", count);
    return count;
}

eddi_gpu_atom_t* eddi_gpu_offload_molecule(eddi_gpu_molecule_t* molecule)
{
    // Allocate memory on the device
    eddi_gpu_atom_t* d_molecule;
    
    check_error(cudaMalloc((void**) &d_molecule, sizeof(eddi_gpu_atom_t) * molecule->n_atoms), "molecule offload - malloc");

    // Then, copy all the atoms on the device
    check_error(cudaMemcpyAsync(d_molecule, molecule->atoms, sizeof(eddi_gpu_atom_t) * molecule->n_atoms, cudaMemcpyHostToDevice, 0), "molecule offload - memcpy");
    printf("Allocating %zu bytes for molecule\n", sizeof(eddi_gpu_atom_t) * molecule->n_atoms );

    return d_molecule;
}

eddi_gpu_molecule_t* eddi_gpu_unload_molecule(eddi_gpu_molecule_t* device_molecule)
{
    eddi_gpu_molecule_t* host_molecule;
    
    host_molecule = (eddi_gpu_molecule_t*) malloc(sizeof(eddi_gpu_molecule_t));
    host_molecule->atoms = (eddi_gpu_atom_t*) malloc(sizeof(eddi_gpu_atom_t) * device_molecule->n_atoms);
    host_molecule->n_atoms = device_molecule->n_atoms;
    if (!host_molecule || ! host_molecule->atoms) 
    {
        EDDI_DEBUG_PRINT("[Error] Could not allocate host molecule\n");
        exit(1);
    }

    check_error(cudaMemcpy(host_molecule->atoms, device_molecule->atoms, sizeof(eddi_gpu_atom_t) * device_molecule->n_atoms, cudaMemcpyDeviceToHost), "cuda memcpy - gpu_unload molecule");
    return host_molecule;
}

void eddi_gpu_free_device_molecule(eddi_gpu_molecule_t* device_molecule)
{
    check_error(cudaFree(device_molecule->atoms), "molecule free - cudafree");
}

void eddi_gpu_free_host_molecule(eddi_gpu_molecule_t* host_molecule)
{
    free(host_molecule->atoms);
}
