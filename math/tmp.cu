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

    clamp(1, 1, 1);
    Floor<float>()(1.0);
    Matrix<float, 10, 10> mat = {{1, 2, 3, 4}, {1, 2, 3, 4}};
}

void test() {
    Vector2D vec;
}
}// namespace vox