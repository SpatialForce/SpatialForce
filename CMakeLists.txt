#  Copyright (c) 2024 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.27)
project(SpatialForce CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

add_subdirectory(core)
add_subdirectory(runtime)