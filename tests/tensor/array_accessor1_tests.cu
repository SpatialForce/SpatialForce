//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "tensor/array.h"
#include "tensor/array_view.h"

#include <gtest/gtest.h>

using namespace vox;

TEST(ArrayView1, Constructors) {
    double data[5];
    for (int i = 0; i < 5; ++i) {
        data[i] = static_cast<double>(i);
    }

    ArrayView1<double> acc(data, 5);

    EXPECT_EQ(5u, acc.length());
    EXPECT_EQ(data, acc.data());
}

TEST(ArrayView1, Iterators) {
    Array1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
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

TEST(ArrayView1, ForEachIndex) {
    Array1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
    auto acc = arr1.view();

    size_t cnt = 0;
    forEachIndex(acc.size(), [&](size_t i) {
        EXPECT_EQ(cnt, i);
        ++cnt;
    });
}

TEST(ConstArrayView1, Constructors) {
    double data[5];
    for (int i = 0; i < 5; ++i) {
        data[i] = static_cast<double>(i);
    }

    // Construct with ArrayView1
    ArrayView1<double> acc(data, 5);
    ConstArrayView1<double> cacc(acc);

    EXPECT_EQ(5u, cacc.length());
    EXPECT_EQ(data, cacc.data());
}

TEST(ConstArrayView1, Iterators) {
    Array1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
    auto acc = arr1.view();

    size_t i = 0;
    for (const float &elem : acc) {
        EXPECT_FLOAT_EQ(acc[i], elem);
        ++i;
    }
}

TEST(ConstArrayView1, ForEachIndex) {
    Array1<float> arr1 = {6.f, 4.f, 1.f, -5.f};
    auto acc = arr1.view();

    size_t cnt = 0;
    forEachIndex(acc.size(), [&](size_t i) {
        EXPECT_EQ(cnt, i);
        ++cnt;
    });
}
