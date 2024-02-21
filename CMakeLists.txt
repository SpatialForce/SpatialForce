cmake_minimum_required(VERSION 3.27)
project(SpatialForce CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(SpatialForce main.cu)

set_target_properties(SpatialForce PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
