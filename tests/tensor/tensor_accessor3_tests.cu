//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "tensor/tensor.h"
#include "tensor/tensor_view.h"

#include <gtest/gtest.h>

using namespace vox;

TEST(TensorView3, Constructors) {
    double data[60];
    for (int i = 0; i < 60; ++i) {
        data[i] = static_cast<double>(i);
    }

    TensorView3<double> acc(data, Vector3UZ(5, 4, 3));

    EXPECT_EQ(5u, acc.size().x);
    EXPECT_EQ(4u, acc.size().y);
    EXPECT_EQ(3u, acc.size().z);
    EXPECT_EQ(data, acc.data());
}

TEST(TensorView3, Iterators) {
    Tensor3<float> arr1(
        {{{1.f, 2.f, 3.f, 4.f},
          {5.f, 6.f, 7.f, 8.f},
          {9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});
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

TEST(TensorView3, ForEachIndex) {
    Tensor3<float> arr1(
        {{{1.f, 2.f, 3.f, 4.f},
          {5.f, 6.f, 7.f, 8.f},
          {9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});
    auto acc = arr1.view();

    forEachIndex(acc.size(), [&](size_t i, size_t j, size_t k) {
        size_t idx = i + (4 * (j + 3 * k)) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), acc(i, j, k));
    });
}

TEST(ConstTensorView3, Constructors) {
    double data[60];
    for (int i = 0; i < 60; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Construct with TensorView3
    TensorView3<double> acc(data, Vector3UZ(5, 4, 3));
    ConstTensorView3<double> cacc(acc);

    EXPECT_EQ(5u, cacc.size().x);
    EXPECT_EQ(4u, cacc.size().y);
    EXPECT_EQ(3u, cacc.size().z);
    EXPECT_EQ(data, cacc.data());
}

TEST(ConstTensorView3, Iterators) {
    Tensor3<float> arr1(
        {{{1.f, 2.f, 3.f, 4.f},
          {5.f, 6.f, 7.f, 8.f},
          {9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});
    auto acc = arr1.view();

    float cnt = 1.f;
    for (const float &elem : acc) {
        EXPECT_FLOAT_EQ(cnt, elem);
        cnt += 1.f;
    }
}

TEST(ConstTensorView3, ForEachIndex) {
    Tensor3<float> arr1(
        {{{1.f, 2.f, 3.f, 4.f},
          {5.f, 6.f, 7.f, 8.f},
          {9.f, 10.f, 11.f, 12.f}},
         {{13.f, 14.f, 15.f, 16.f},
          {17.f, 18.f, 19.f, 20.f},
          {21.f, 22.f, 23.f, 24.f}}});
    auto acc = arr1.view();

    forEachIndex(acc.size(), [&](size_t i, size_t j, size_t k) {
        size_t idx = i + (4 * (j + 3 * k)) + 1;
        EXPECT_FLOAT_EQ(static_cast<float>(idx), acc(i, j, k));
    });
}
