#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <limits.h>

#include "eddi_molecule_readers.h"


void convert_file(const char* input_file_name, const char* output_file_name)
{
    bool first = true;
    eddi_molecule_t molecule;

    eddi_read_pdb(input_file_name, &molecule);
    FILE* output_file;

    float min_x = 0, min_y = 0, min_z = 0;
    float max_x = 0, max_y = 0, max_z = 0;

    if (!(output_file = fopen(output_file_name, "w")))
    {
        perror("Error opening output file");
        eddi_free_molecule(&molecule);
        eddi_free_atomic_data();
        return;
    }

    fprintf(output_file, "%zu\n", molecule.n_atoms);
    for (size_t i = 0; i < molecule.n_atoms; i++)
    {
        if (first)
        {
            min_x = molecule.atoms_x[i];
            min_y = molecule.atoms_y[i];
            min_z = molecule.atoms_z[i];
            max_x = molecule.atoms_x[i];
            max_y = molecule.atoms_y[i];
            max_z = molecule.atoms_z[i];
            first = false;
        }

        fprintf(output_file, "%f %f %f %d\n",
            molecule.atoms_x[i],
            molecule.atoms_y[i],
            molecule.atoms_z[i],
            molecule.atomic_numbers[i]);

        if (molecule.atoms_x[i] < min_x) min_x = molecule.atoms_x[i];
        if (molecule.atoms_y[i] < min_y) min_y = molecule.atoms_y[i];
        if (molecule.atoms_z[i] < min_z) min_z = molecule.atoms_z[i];

        if (molecule.atoms_x[i] > max_x) max_x = molecule.atoms_x[i];
        if (molecule.atoms_y[i] > max_y) max_y = molecule.atoms_y[i];
        if (molecule.atoms_z[i] > max_z) max_z = molecule.atoms_z[i];
    }

    printf("-> Min coordinates: (%f, %f, %f)\n", min_x, min_y, min_z);
    printf("-> Max coordinates: (%f, %f, %f)\n", max_x, max_y, max_z);

    fclose(output_file);
    eddi_free_molecule(&molecule);
    eddi_free_atomic_data();
}

void convert_bundle(const char* input_dir, const char* output_file_name)
{
    bool first = true;
    size_t atoms_number = 0;
    eddi_molecule_t* molecule;


    DIR* dir;
    struct dirent* entry;
    char file_path[PATH_MAX];
    FILE* output_file;

    float min_x = 0, min_y = 0, min_z = 0;
    float max_x = 0, max_y = 0, max_z = 0;

    if (!(dir = opendir(input_dir)))
    {
        perror("opendir");
        return;
    }

    printf("Opened dir %s\n", input_dir);

    if (!(output_file = fopen(output_file_name, "w")))
    {
        perror("fopen");
        closedir(dir);
        return;
    }

    size_t pdb_file_count = 0;

    while ((entry = readdir(dir)) != NULL)
    {
        if (strstr(entry->d_name, ".pdb"))
        {
            pdb_file_count++;
        }
    }

    char** file_names = malloc(sizeof(char*) * pdb_file_count);
    if (!file_names)
    {
        perror("malloc");
        closedir(dir);
        fclose(output_file);
        return;
    }

    size_t file_index = 0;
    rewinddir(dir); // Reset directory stream for further processing

    while ((entry = readdir(dir)) != NULL)
    {
        if (strstr(entry->d_name, ".pdb"))
        {
            file_names[file_index] = malloc(PATH_MAX);
            if (!file_names[file_index])
            {
                perror("malloc");
                for (size_t j = 0; j < file_index; ++j)
                {
                    free(file_names[j]);
                }
                free(file_names);
                closedir(dir);
                fclose(output_file);
                return;
            }
            snprintf(file_names[file_index], PATH_MAX, "%s/%s", input_dir, entry->d_name);
            file_index++;
        }
    }
    closedir(dir);
    
    printf("Number of .pdb files: %zu\n", pdb_file_count);

    molecule = malloc(sizeof(eddi_molecule_t) * pdb_file_count);

    size_t current_molecule = 0;

    #pragma omp parallel for
    for (size_t file_idx = 0; file_idx < pdb_file_count; ++file_idx)
    {
        eddi_read_pdb(file_names[file_idx], &molecule[file_idx]);
        printf("Entry: %s -> %zu atoms\n", file_names[file_idx], molecule[file_idx].n_atoms);
        file_idx++;
    }

    for(size_t mol_idx = 0; mol_idx < pdb_file_count; ++mol_idx)
    {
        atoms_number += molecule[mol_idx].n_atoms;
    }


    fprintf(output_file, "%zu\n", atoms_number);

    for (size_t mol_idx = 0; mol_idx < pdb_file_count; ++mol_idx)
    {
        for (size_t i = 0; i < molecule[mol_idx].n_atoms; ++i)
        {
            if (first)
            {
            min_x = molecule[mol_idx].atoms_x[i];
            min_y = molecule[mol_idx].atoms_y[i];
            min_z = molecule[mol_idx].atoms_z[i];
            max_x = molecule[mol_idx].atoms_x[i];
            max_y = molecule[mol_idx].atoms_y[i];
            max_z = molecule[mol_idx].atoms_z[i];
            first = false;
            }

            fprintf(output_file, "%f %f %f %d\n",
            molecule[mol_idx].atoms_x[i],
            molecule[mol_idx].atoms_y[i],
            molecule[mol_idx].atoms_z[i],
            molecule[mol_idx].atomic_numbers[i]);

            if (molecule[mol_idx].atoms_x[i] < min_x) min_x = molecule[mol_idx].atoms_x[i];
            if (molecule[mol_idx].atoms_y[i] < min_y) min_y = molecule[mol_idx].atoms_y[i];
            if (molecule[mol_idx].atoms_z[i] < min_z) min_z = molecule[mol_idx].atoms_z[i];

            if (molecule[mol_idx].atoms_x[i] > max_x) max_x = molecule[mol_idx].atoms_x[i];
            if (molecule[mol_idx].atoms_y[i] > max_y) max_y = molecule[mol_idx].atoms_y[i];
            if (molecule[mol_idx].atoms_z[i] > max_z) max_z = molecule[mol_idx].atoms_z[i];
        }
        eddi_free_molecule(&molecule[mol_idx]);
    }

    printf("-> Total atoms: %zu\n", atoms_number);
    printf("-> Min coordinates: (%f, %f, %f)\n", min_x, min_y, min_z);
    printf("-> Max coordinates: (%f, %f, %f)\n", max_x, max_y, max_z);

    free(molecule);
    fclose(output_file);
    
    eddi_free_atomic_data();
}

void print_help(const char* program_name)
{
    printf("pdb_to_coord converts a pdb file or bundle to a coordinate (Bohrs) + atomic_number file:\n\n"
           "NUMBER_OF_ATOMS\n"
           "x_i y_i z_i atomic_number_i\n\n");
    printf("To convert a file\n\n %s file <path_to_file> <output_path>\n\n", program_name);
    printf("To convert a bundle of files\n\n %s bundle <path_to_dir> <output_path>\n\n", program_name);
}

int main(int argc, char** argv)
{

    if (argc != 4)
    {
        print_help(argv[0]);
        return 0;
    }


    if (strcmp(argv[1], "file") == 0)
    {
        if (access(argv[2], F_OK) != 0)
        {
            printf("Error: %s is not a valid file.\n", argv[2]);
            return 1;
        }
        convert_file(argv[2], argv[3]);

    } else if (strcmp(argv[1], "bundle") == 0)
    {
        struct stat sb;
        if (stat(argv[2], &sb) != 0 || !S_ISDIR(sb.st_mode) || (opendir(argv[2]) && readdir(opendir(argv[2])) == NULL))
        {
            printf("Error: %s is not a valid directory.\n", argv[2]);
            return 1;
        }
        convert_bundle(argv[2], argv[3]);
    }


    return 0;
}