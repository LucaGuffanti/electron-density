message(STATUS "=== Configuring EDDI ===")

# Collect all source files

file(GLOB_RECURSE SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
file(GLOB_RECURSE GPU_C_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/gpu/*.c")

# Print the list of source files
foreach(SOURCE_FILE ${SOURCES})
    message(STATUS "Source file: ${SOURCE_FILE}")
endforeach()

# Print the list of GPU C source files
foreach(GPU_SOURCE_FILE ${GPU_C_SOURCES})
    message(STATUS "GPU C source file: ${GPU_SOURCE_FILE}")
endforeach()

# Check for CUDA


if (EDDI_CUDA_FOUND)
message(STATUS "CUDA found, searching for .cu files")
    file(GLOB_RECURSE CUDA_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

    list(APPEND SOURCES ${CUDA_SOURCES})

    # Print the list of source files
    foreach(SOURCE_FILE ${SOURCES})
    message(STATUS "Source file: ${SOURCE_FILE}")
    endforeach()
    
    # Enable separable compilation for CUDA
else()
    list(REMOVE_ITEM SOURCES ${GPU_C_SOURCES})
    message(STATUS "CUDA not found, skipping .cu files")
endif()

# Create shared or static library
if (BUILD_SHARED_LIB)
    message(STATUS "Configuring EDDI as a shared library")
    add_library(EDDI SHARED ${SOURCES})

    # Set CUDA properties for shared library
    if (EDDI_CUDA_FOUND)
        set_target_properties(EDDI PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        set_target_properties(EDDI PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
        set_target_properties(EDDI PROPERTIES CUDA_ARCHITECTURES "80;86")
        set_target_properties(EDDI PROPERTIES POSITION_INDEPENDENT_CODE ON)

        # Suppressing deprecated warning for GPU 
        set_target_properties(EDDI PROPERTIES CUDA_PROPAGATE_HOST_FLAGS OFF)
        target_compile_options(EDDI PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -diag-suppress 2464>)
        
        message(STATUS "Linking CUDA Toolkit libraries")
        find_package(CUDAToolkit REQUIRED)
        target_link_libraries(EDDI PUBLIC CUDA::cudart)
    endif()
    # Set shared library properties

else()
    if (EDDI_CUDA_FOUND)
        message(FATAL_ERROR "Cannot build the CUDA part as a static library. Please set BUILD_SHARED_LIB to ON.")
    endif()
    message(STATUS "Configuring EDDI as a static library")
    add_library(EDDI STATIC ${SOURCES})
endif()

# Enable OpenMP if requested
if (ENABLE_OPENMP)
    message(STATUS "Configuring EDDI with OpenMP")
    find_package(OpenMP REQUIRED)
    target_link_libraries(EDDI PUBLIC OpenMP::OpenMP_C)
endif()

# Debug mode settings
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Define debug macro
    message(STATUS "Configuring EDDI in DEBUG mode")
    target_compile_definitions(EDDI PRIVATE EDDI_DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
    # Profile mode settings
    message(STATUS "Configuring EDDI in PROFILE mode")
else()
    # Release mode settings
    message(STATUS "Configuring EDDI in RELEASE mode")
endif()

# Choice of Wavefunctions
if (DENSITY_FUNCTION STREQUAL "Slater")
    message(STATUS "Configuring EDDI with Slater wavefunctions")
    target_compile_definitions(EDDI PRIVATE EDDI_SLATER_WF)
elseif(DENSITY_FUNCTION STREQUAL "Clementi")
    message(STATUS "Configuring EDDI with Clementi wavefunctions")
    target_compile_definitions(EDDI PRIVATE EDDI_CLEMENTI_WF)
else()
    message(FATAL_ERROR "Invalid density function choice. Please choose 'Slater' or 'Clementi'.")
endif()

# Include directories
if (EDDI_CUDA_FOUND)
    target_include_directories(EDDI PUBLIC ${EDDI_GPU_INCLUDE_DIRECTORIES})
else()
    target_include_directories(EDDI PUBLIC ${EDDI_INCLUDE_DIRECTORIES})
endif()
# Link math library
target_link_libraries(EDDI PRIVATE m)

message(STATUS "=== EDDI configured ===")
