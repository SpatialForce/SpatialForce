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

set(SRC
        main.cpp
        ${RUNTIME_TESTS}
)

add_executable(${PROJECT_NAME} ${SRC})

find_package(GTest CONFIG REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE runtime GTest::gtest GTest::gtest_main GTest::gmock GTest::gmock_main CUDA::cuda_driver)

target_include_directories(${PROJECT_NAME} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR} .
        ../)
