//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "tensor/tensor.h"
#include "tensor/tensor_view.h"

#include <gtest/gtest.h>

using namespace vox;

TEST(ConstTensorView, Constructors) {
    Tensor2<double> arr = {{1, 2}, {3, 4}, {5, 6}};
    TensorView2<double> view = arr.view();

    // Copy from mutable Tensor
    ConstTensorView2<double> view2(arr);
    EXPECT_EQ(Vector2UZ(2, 3), view2.shape());
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(arr(i, j), view2(i, j));
        }
    }

    // Copy from mutable TensorView
    ConstTensorView2<double> view3(arr.view());
    EXPECT_EQ(Vector2UZ(2, 3), view3.shape());
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(arr(i, j), view3(i, j));
        }
    }

    // Copy from immutable TensorView
    ConstTensorView2<double> view4(view3);
    EXPECT_EQ(Vector2UZ(2, 3), view4.shape());
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 2; ++i) {
            EXPECT_DOUBLE_EQ(arr(i, j), view4(i, j));
        }
    }
}
