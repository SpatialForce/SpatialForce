//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "runtime/device.h"

using namespace vox;

TEST(name, Device) {
    vox::init();

    std::cout << "total cuda device: " << device_count() << std::endl;
    std::cout << device(0).name() << std::endl;
    std::cout << device(0).arch() << std::endl;
}