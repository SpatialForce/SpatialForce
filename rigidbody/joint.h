//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/matrix.h"

namespace vox {
// Types of joints linking rigid bodies
enum class JointType {
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    JOINT_BALL,
    JOINT_FIXED,
    JOINT_FREE,
    JOINT_COMPOUND,
    JOINT_UNIVERSAL,
    JOINT_DISTANCE,
    JOINT_D6,
};

// Joint axis mode types
enum class JointMode {
    LIMIT,
    TARGET_POSITION,
    TARGET_VELOCITY
};

/// Describes a joint axis that can have limits and be driven towards a target.
struct JointAxis {
    /// The 3D axis that this JointAxis object describes
    Vector3F axis;
    /// The lower limit of the joint axis
    float limit_lower;
    //// The upper limit of the joint axis
    float limit_upper;
    /// The elastic stiffness of the joint axis limits
    float limit_ke = 100.0;
    /// The damping stiffness of the joint axis limits
    float limit_kd = 10.0;
    /// The target position or velocity (depending on the mode, see `Joint modes`_) of the joint axis
    float target{};
    /// The proportional gain of the joint axis target drive PD controller
    float target_ke = 0.0;
    /// The derivative gain of the joint axis target drive PD controller
    float target_kd = 0.0;
    /// The mode of the joint axis
    JointMode mode = JointMode::TARGET_POSITION;

    explicit JointAxis(const Vector3F &axis);
};
}// namespace vox