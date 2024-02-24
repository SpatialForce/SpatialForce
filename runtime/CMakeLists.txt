#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(runtime LANGUAGES C CXX CUDA)

set(COMMON_FILES
        cuda_util.h
        cuda_util.cu
        device.h
        device.cu
        event.h
        event.cu
        stream.h
        stream.cu
        graph.h
        graph.cu
)

set(RESOURCE_FILES
        cuda_buffer.h
        cuda_texture.h
        cuda_tensor.h
        cuda_tensor_view.h
)

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
        ${RESOURCE_FILES}
)

add_library(${PROJECT_NAME} OBJECT ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} .
        ../)