#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

if (NOT DEFINED CMAKE_C_COMPILER_LAUNCHER AND NOT DEFINED CMAKE_CXX_COMPILER_LAUNCHER)
    find_program(CCACHE_FOUND ccache)
    find_program(SCCACHE_FOUND sccache)
    if (CCACHE_FOUND AND NOT SCCACHE_FOUND)
        message("setting CCACHE to ${CCACHE_FOUND}")
        set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_FOUND})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_FOUND})
    elseif (SCCACHE_FOUND AND NOT CCACHE_FOUND)
        message("setting CCACHE to ${CCACHE_FOUND}")
        set(CMAKE_C_COMPILER_LAUNCHER ${SCCACHE_FOUND})
        set(CMAKE_CXX_COMPILER_LAUNCHER ${SCCACHE_FOUND})
    endif (CCACHE_FOUND AND NOT SCCACHE_FOUND)
endif ()

cmake_minimum_required(VERSION 3.16)

project(SpatialForce)

# Add path for local cmake scripts
list(APPEND CMAKE_MODULE_PATH
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

include(global_options)
include(check_atomic)

# CUDA
find_package(CUDA)
if (CUDA_FOUND)
    message("Found CUDA")
    find_package(CUDAToolkit)
    set(CMAKE_CUDA_ARCHITECTURES 86)
    enable_language(CUDA)
    include_directories(${CUDA_INCLUDE_DIRS})
endif ()

add_subdirectory(third_party)

# Add third party libraries
add_subdirectory(core)
add_subdirectory(runtime)
add_subdirectory(rigidbody)
add_subdirectory(fields)

add_subdirectory(unittest)
add_subdirectory(benchmark)