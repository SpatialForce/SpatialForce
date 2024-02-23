#  Copyright (c) 2023 Feng Yang
#
#  I am making my contributions/submissions to this project solely in my
#  personal capacity and am not conveying any rights to any intellectual
#  property of any third parties.

# create sample app project
project(sim-tests LANGUAGES C CXX CUDA)

set(RUNTIME_TESTS
        runtime/device_tests.cu
        runtime/buffer_tests.cu
)

set(MATH_TESTS
        math/math_utils_tests.cu
        math/vector2_tests.cu
        math/vector3_tests.cu
        math/vector4_tests.cu
        math/vector_tests.cu
        math/matrix2x2_tests.cu
        math/matrix3x3_tests.cu
        math/matrix4x4_tests.cu
        math/matrix_tests.cu
)

set(FIELDS_TESTS
        fields/compile_tests.cu
        fields/volume_integrator_tests.cu
)

set(SRC
        cuda_gtest_plugin.h
        ${RUNTIME_TESTS}
        ${MATH_TESTS}
        ${FIELDS_TESTS}
)

add_executable(${PROJECT_NAME} ${SRC})

find_package(GTest CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE
        runtime fields
        GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main
        CUDA::cuda_driver
)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} .
        ../)
