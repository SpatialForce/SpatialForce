//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "runtime/device.h"

using namespace vox;

TEST(name, Device) {
    std::cout << device(0).name << std::endl;
    std::cout << device(0).arch << std::endl;

    // device(index) will init cuda, otherwise count will be 0.
    std::cout << "total cuda device: " << device_count() << std::endl;
}