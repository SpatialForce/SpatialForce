//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "bounding_box.h"
#include "intersect.h"

namespace vox {
struct BVHPackedNodeHalf {
    float x;
    float y;
    float z;
    unsigned int i : 31;
    unsigned int b : 1;
};
}// namespace vox