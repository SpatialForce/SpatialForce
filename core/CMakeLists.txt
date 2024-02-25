#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(core LANGUAGES C CXX CUDA)

set(COMMON_FILES
        define.h
)

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
)

add_library(${PROJECT_NAME} INTERFACE ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR} .
        ../)