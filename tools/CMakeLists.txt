# Collect all test source files
file(GLOB_RECURSE UTILITY_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

# Macro to configure individual tests
macro(eddi_utility_prepare UTILITY_SOURCE)
    get_filename_component(UTILITY_NAME ${UTILITY_SOURCE} NAME_WE)
    add_executable(${UTILITY_NAME} ${UTILITY_SOURCE})

    # Link the utility with the EDDI library and math library
    target_link_libraries(${UTILITY_NAME} EDDI m)

    # Include necessary directories
    target_include_directories(${UTILITY_NAME} PUBLIC ${EDDI_INCLUDE_DIRECTORIES})

    # Apply AddressSanitizer flags for Debug builds (only for GCC/Clang)
    if (CMAKE_BUILD_TYPE MATCHES "Debug")
        target_compile_definitions(${UTILITY_NAME} PRIVATE EDDI_DEBUG)
    endif()
endmacro()

# Macro to add a basic utility
macro(add_eddi_utility UTILITY_SOURCE)
    message(STATUS "Adding utility ${UTILITY_SOURCE}")
    eddi_utility_prepare(${UTILITY_SOURCE})
    add_test(NAME ${UTILITY_NAME} COMMAND ${UTILITY_NAME} ${ARGN})
endmacro()


add_eddi_utility(pdb_to_coord.c)