//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "runtime/buffer.h"

using namespace vox;

TEST(Buffer, constructor) {
    vox::init();

    auto buffer = create_buffer<float>(10);
    std::vector<float> a{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    sync_h2d(a.data(), buffer);
    std::vector<float> result(10);
    sync_d2h(buffer, result.data());
    EXPECT_EQ(result[0], 1);
}