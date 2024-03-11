#  Copyright (c) 2024 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.27)
project(SpatialForce C CXX CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 89)

find_package(CUDAToolkit)
set(GMSH_LIB ~/Desktop/gmsh/lib)
set(GMSH_INCLUDE ~/Desktop/gmsh/include)

add_subdirectory(core)

add_subdirectory(math)
add_subdirectory(shape)
add_subdirectory(tensor)
add_subdirectory(runtime)
add_subdirectory(rigidbody)

add_subdirectory(fields)

add_subdirectory(tests)