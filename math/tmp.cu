//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "matrix.h"

namespace vox {
__global__ void kernel_test() {
    Vector2D vec;
    Vector2D vec2;
    vec2 += vec;
}

void test() {
    Vector2D vec;
}
}// namespace vox