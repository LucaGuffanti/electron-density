cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0048 NEW)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0104 NEW)

# Set project version
project(EDDI VERSION 1.0.0 LANGUAGES C CUDA)


# Set C standards
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# Option to build tests
option(BUILD_SHARED_LIB "Build EDDI as a shared library" ON)
option(ENABLE_OPENMP "Enable OpenMP support" ON)
option(ENABLE_FASTMATH "Enable fast math optimizations" OFF)
option(BUILD_TESTS "Build the tests" ON)
option(DENSITY_FUNCTION "Choose the density function (Slater, Clementi)" ON)

if(NOT DENSITY_FUNCTION)
    set(DENSITY_FUNCTION "Slater" CACHE STRING "Choose the density function (Slater, Clementi)" FORCE)
endif()

set_property(CACHE DENSITY_FUNCTION PROPERTY STRINGS "Slater" "Clementi")

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build (Debug, Release, Profile)." FORCE)
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "Profile")
endif()


find_package(CUDA QUIET)
set(EDDI_CUDA_FOUND ${CUDA_FOUND})


# Eddi Library (CPU-Based with OpenMP)
add_subdirectory(eddi)

if (ENABLE_FASTMATH)
    message(STATUS "Fast math optimizations enabled")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
endif()

# Debug mode configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Debug mode activated")

    # Enable debug definitions
    add_compile_definitions(EDDI_DEBUG)

    # AddressSanitizer settings
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address -fno-omit-frame-pointer -O0 -g -Wall -Wextra")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address -fno-omit-frame-pointer")
    set(CMAKE_LINKER_FLAGS "${CMAKE_LINKER_FLAGS} -fsanitize=address")

    # Set environment variable to catch memory leaks
    set(ENV{ASAN_OPTIONS} "detect_leaks=1")

    message(STATUS "AddressSanitizer enabled for memory leak detection")

elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
    message(STATUS "Profiling mode activated")

    # Profiling mode configuration
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g -O3 -Wall -Wextra -g")
    if(EDDI_CUDA_FOUND)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -O3")
    endif()

else()
    message(STATUS "Release mode activated")
    # Release mode configuration
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
    
endif()


# Add tests if BUILD_TESTS is ON
if(BUILD_TESTS)
    message(STATUS "=== Configuring Tests ===")
    enable_testing()
    add_subdirectory(test)
    message(STATUS "=== Tests configured ===")
endif()

add_subdirectory(tools)