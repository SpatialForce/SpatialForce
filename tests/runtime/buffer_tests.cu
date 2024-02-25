//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "math/matrix.h"
#include "runtime/cuda_buffer.h"
#include "runtime/cuda_texture.h"
#include "runtime/cuda_tensor.h"

using namespace vox;

TEST(Buffer, raw) {
    vox::init();

    CudaBuffer<float> buffer;
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    buffer = a;
    std::vector<float> result;
    buffer.copyTo(result);
    EXPECT_EQ(result[0], 1);
}

TEST(Buffer, host_device) {
    vox::init();

    auto host_device = HostDeviceVector<float>();
    host_device.host_buffer = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    host_device.sync_h2d();

    std::vector<float> result(10);
    host_device.device_buffer.copyTo(result);
    EXPECT_EQ(result[0], 1);
}

TEST(Buffer, view) {
    vox::init();

    auto host_device = HostDeviceVector<float>();
    host_device.host_buffer = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    host_device.sync_h2d();
}