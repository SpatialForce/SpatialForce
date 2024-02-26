//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "math/ray.h"
#include "cuda_gtest_plugin.h"
#include <gtest/gtest.h>

using namespace vox;

CUDA_TEST(Ray3, Constructors) {
    Ray3D ray;
    EXPECT_EQ(Vector3D(), ray.origin);
    EXPECT_EQ(Vector3D(1, 0, 0), ray.direction);

    Ray3D ray2(Vector3D(1, 2, 3), Vector3D(4, 5, 6));
    EXPECT_EQ(Vector3D(1, 2, 3), ray2.origin);
    EXPECT_EQ(Vector3D(4, 5, 6).normalized(), ray2.direction);

    Ray3D ray3(ray2);
    EXPECT_EQ(Vector3D(1, 2, 3), ray3.origin);
    EXPECT_EQ(Vector3D(4, 5, 6).normalized(), ray3.direction);
}

CUDA_TEST(Ray3, PointAt) {
    Ray3D ray;
    EXPECT_EQ(Vector3D(4.5, 0.0, 0.0), ray.pointAt(4.5));
}