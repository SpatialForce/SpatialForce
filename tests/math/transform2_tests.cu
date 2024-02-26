//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "tests_utils.h"

#include "math/transform.h"

#include <gtest/gtest.h>

using namespace vox;

CUDA_TEST(Transform2, Constructors) {
    Transform2 t1;

    EXPECT_EQ(Vector2F(), t1.translation());
    EXPECT_EQ(0.0, t1.orientation().rotation());

    Transform2 t2({2.0, -5.0}, kQuarterPiD);

    EXPECT_EQ(Vector2F(2.0, -5.0), t2.translation());
    EXPECT_NEAR(kQuarterPiD, t2.orientation().rotation(), 1.0e-7);
}

CUDA_TEST(Transform2, Transform) {
    Transform2 t({2.0, -5.0}, kHalfPiD);

    auto r1 = t.toWorld({4.0, 1.0});
    EXPECT_NEAR(1.0, r1.x, 1.0e-5);
    EXPECT_NEAR(-1.0, r1.y, 1.0e-5);

    auto r2 = t.toLocal(r1);
    EXPECT_NEAR(4.0, r2.x, 1.0e-5);
    EXPECT_NEAR(1.0, r2.y, 1.0e-5);

    auto r3 = t.toWorldDirection({4.0, 1.0});
    EXPECT_NEAR(-1.0, r3.x, 1.0e-5);
    EXPECT_NEAR(4.0, r3.y, 1.0e-5);

    auto r4 = t.toLocalDirection(r3);
    EXPECT_NEAR(4.0, r4.x, 1.0e-5);
    EXPECT_NEAR(1.0, r4.y, 1.0e-5);

    BoundingBox2F bbox({-2, -1}, {2, 1});
    auto r5 = t.toWorld(bbox);
    EXPECT_BOUNDING_BOX2_NEAR(BoundingBox2F({1, -7}, {3, -3}), r5, 1.0e-5);

    auto r6 = t.toLocal(r5);
    EXPECT_BOUNDING_BOX2_NEAR(bbox, r6, 1.0e-5);
}
