//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "runtime/buffer.h"

using namespace vox;

TEST(Buffer, raw) {
    vox::init();

    auto buffer = create_buffer<float>();
    buffer.alloc(10);
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    sync_h2d(a.data(), buffer);
    std::vector<float> result(10);
    sync_d2h(buffer, result.data());
    EXPECT_EQ(result[0], 1);
}

TEST(Buffer, host_device) {
    vox::init();

    auto host_device = HostDeviceBuffer<float>();
    host_device.host_buffer = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    host_device.sync_h2d();

    std::vector<float> result(10);
    sync_d2h(host_device.device_buffer, result.data());
    EXPECT_EQ(result[0], 1);
}

TEST(Buffer, view) {
    vox::init();

    auto host_device = HostDeviceBuffer<float>();
    host_device.host_buffer = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    host_device.sync_h2d();
    auto view = host_device.view();

    EXPECT_EQ(view.ndim, 1);
}