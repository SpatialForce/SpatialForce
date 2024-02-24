//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "tensor/tensor.h"
#include "tensor/tensor_view.h"

#include <gtest/gtest.h>

using namespace vox;

TEST(TensorView2, Constructors) {
    double data[20];
    for (int i = 0; i < 20; ++i) {
        data[i] = static_cast<double>(i);
    }

    TensorView2<double> acc(data, Vector2UZ(5, 4));

    EXPECT_EQ(5u, acc.shape().x);
    EXPECT_EQ(4u, acc.shape().y);
    EXPECT_EQ(data, acc.data());
}

TEST(TensorView2, Iterators) {
    Tensor2<float> arr1(
        {{1.f, 2.f, 3.f, 4.f},
         {5.f, 6.f, 7.f, 8.f},
         {9.f, 10.f, 11.f, 12.f}});
    auto acc = arr1.view();

    float cnt = 1.f;
    for (float &elem : acc) {
        EXPECT_FLOAT_EQ(cnt, elem);
        cnt += 1.f;
    }

    cnt = 1.f;
    for (const float &elem : acc) {
        EXPECT_FLOAT_EQ(cnt, elem);
        cnt += 1.f;
    }
}

TEST(TensorView2, ForEachIndex) {
    Tensor2<float> arr1(
        {{1.f, 2.f, 3.f, 4.f},
         {5.f, 6.f, 7.f, 8.f},
         {9.f, 10.f, 11.f, 12.f}});

    forEachIndex(arr1.shape(), [&](size_t i, size_t j) {
        size_t idx = i + (4 * j) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), arr1(i, j));
    });
}

TEST(ConstTensorView2, Constructors) {
    double data[20];
    for (int i = 0; i < 20; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Construct with TensorView2
    TensorView2<double> acc(data, Vector2UZ(5, 4));
    ConstTensorView2<double> cacc(acc);

    EXPECT_EQ(5u, cacc.shape().x);
    EXPECT_EQ(4u, cacc.shape().y);
    EXPECT_EQ(data, cacc.data());
}

TEST(ConstTensorView2, Iterators) {
    Tensor2<float> arr1(
        {{1.f, 2.f, 3.f, 4.f},
         {5.f, 6.f, 7.f, 8.f},
         {9.f, 10.f, 11.f, 12.f}});
    auto acc = arr1.view();

    float cnt = 1.f;
    for (const float &elem : acc) {
        EXPECT_FLOAT_EQ(cnt, elem);
        cnt += 1.f;
    }
}

TEST(ConstTensorView2, ForEach) {
    Tensor2<float> arr1(
        {{1.f, 2.f, 3.f, 4.f},
         {5.f, 6.f, 7.f, 8.f},
         {9.f, 10.f, 11.f, 12.f}});
    auto acc = arr1.view();

    size_t i = 0;
    std::for_each(acc.begin(), acc.end(), [&](float val) {
        EXPECT_FLOAT_EQ(acc[i], val);
        ++i;
    });
}

TEST(ConstTensorView2, ForEachIndex) {
    Tensor2<float> arr1(
        {{1.f, 2.f, 3.f, 4.f},
         {5.f, 6.f, 7.f, 8.f},
         {9.f, 10.f, 11.f, 12.f}});
    auto acc = arr1.view();

    forEachIndex(acc.shape(), [&](size_t i, size_t j) {
        size_t idx = i + (4 * j) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), acc(i, j));
    });
}
