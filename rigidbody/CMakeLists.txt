#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

cmake_minimum_required(VERSION 3.16)

project(rigidbody LANGUAGES C CXX CUDA)

set(COMMON_FILES
        utils.h
        collide.h
        collide.cu
        inertia.h
        inertia.cu
        articulation.h
        articulation.cu
        model.h
        model.cu
        model_builder.h
        model_builder.cpp
        optimizer.h
        optimizer.cu
        particles.h
        particles.cu
        integrator_euler.h
        integrator_euler.cu
        integrator_xpbd.h
        integrator_xpbd.cu
)

source_group("common\\" FILES ${COMMON_FILES})

set(PROJECT_FILES
        ${COMMON_FILES}
)

add_library(${PROJECT_NAME} OBJECT ${PROJECT_FILES})
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} .
        ../)

if (${NEED_LINK_ATOMIC})
    target_link_libraries(${PROJECT_NAME} PUBLIC atomic)
endif ()

if (CMAKE_DO_CLANG_TIDY)
    set_target_properties(framework PROPERTIES CXX_CLANG_TIDY "${CMAKE_DO_CLANG_TIDY}")
endif ()