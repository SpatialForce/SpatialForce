//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "math/matrix.h"
#include "runtime/cuda_tensor.h"

using namespace vox;

TEST(Tensor, raw) {
    vox::init();

    CudaTensor1<float> tensor;
    tensor.resize(100);
}