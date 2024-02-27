//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "joint.h"

namespace vox {
JointAxis::JointAxis(const Vector3F &axis)
    : axis(axis.normalized()) {
    limit_lower = -thrust::numeric_limits<float>::infinity();
    limit_upper = thrust::numeric_limits<float>::infinity();
}

}// namespace vox