//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "runtime/device.h"

using namespace vox;

TEST(name, Device) {
    std::cout << device_info(0).name << std::endl;
}