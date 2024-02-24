//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "tensor/tensor.h"
#include "tensor/tensor_view.h"

#include <gtest/gtest.h>

using namespace vox;

TEST(TensorView1, Constructors) {
    double data[5];
    for (int i = 0; i < 5; ++i) {
        data[i] = static_cast<double>(i);
    }

    TensorView1<double> acc(data, 5);

    EXPECT_EQ(5u, acc.length());
    EXPECT_EQ(data, acc.data());
}

TEST(TensorView1, Iterators) {
    Tensor1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
    auto acc = arr1.view();

    size_t i = 0;
    for (float &elem : acc) {
        EXPECT_FLOAT_EQ(acc[i], elem);
        ++i;
    }

    i = 0;
    for (const float &elem : acc) {
        EXPECT_FLOAT_EQ(acc[i], elem);
        ++i;
    }
}

TEST(TensorView1, ForEachIndex) {
    Tensor1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
    auto acc = arr1.view();

    size_t cnt = 0;
    forEachIndex(acc.shape(), [&](size_t i) {
        EXPECT_EQ(cnt, i);
        ++cnt;
    });
}

TEST(ConstTensorView1, Constructors) {
    double data[5];
    for (int i = 0; i < 5; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Construct with TensorView1
    TensorView1<double> acc(data, 5);
    ConstTensorView1<double> cacc(acc);

    EXPECT_EQ(5u, cacc.length());
    EXPECT_EQ(data, cacc.data());
}

TEST(ConstTensorView1, Iterators) {
    Tensor1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
    auto acc = arr1.view();

    size_t i = 0;
    for (const float &elem : acc) {
        EXPECT_FLOAT_EQ(acc[i], elem);
        ++i;
    }
}

TEST(ConstTensorView1, ForEachIndex) {
    Tensor1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
    auto acc = arr1.view();

    size_t cnt = 0;
    forEachIndex(acc.shape(), [&](size_t i) {
        EXPECT_EQ(cnt, i);
        ++cnt;
    });
}
