//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "joint.h"
#include "model_builder.h"

namespace vox {
JointAxis::JointAxis(const Vector3F &axis)
    : axis(axis.normalized()) {
    limit_lower = -thrust::numeric_limits<float>::infinity();
    limit_upper = thrust::numeric_limits<float>::infinity();
}

JointModelBuilder::JointModelBuilder(ModelBuilder &builder) : builder{builder} {
}

size_t JointModelBuilder::joint_count() {
    return joint_type.width();
}

size_t JointModelBuilder::joint_axis_count() {
    return joint_axis.width();
}

size_t JointModelBuilder::articulation_count() {
    return articulation_start.width();
}

void JointModelBuilder::add_articulation() {
    articulation_start.append(joint_count());
}

void JointModelBuilder::_add_axis_dim(const JointAxis &dim) {
    joint_axis.append(dim.axis);
    joint_axis_mode.append(dim.mode);
    joint_target.append(dim.target);
    joint_target_ke.append(dim.target_ke);
    joint_target_kd.append(dim.target_kd);
    joint_limit_ke.append(dim.limit_ke);
    joint_limit_kd.append(dim.limit_kd);
    if (std::isfinite(dim.limit_lower)) {
        joint_limit_lower.append(dim.limit_lower);
    } else {
        joint_limit_lower.append(-1e6);
    }
    if (std::isfinite(dim.limit_upper)) {
        joint_limit_upper.append(dim.limit_upper);
    } else {
        joint_limit_upper.append(1e6);
    }
}

size_t JointModelBuilder::add_joint(JointType type,
                                    int parent,
                                    int child,
                                    std::initializer_list<JointAxis> linear_axes,
                                    std::initializer_list<JointAxis> angular_axes,
                                    std::optional<std::string_view> name,
                                    const TransformF &parent_xform,
                                    const TransformF &child_xform,
                                    float linear_compliance,
                                    float angular_compliance,
                                    bool collision_filter_parent,
                                    bool enabled) {
    if (articulation_start.width() == 0)
        // automatically add an articulation if none exists
        add_articulation();
    joint_type.append(joint_type);
    joint_parent.append(parent);
    if (auto iter = joint_parents.find(child); iter == joint_parents.end()) {
        joint_parents[child] = {parent};
    } else {
        iter->second.append(parent);
    }
    joint_child.append(child);
    joint_X_p.append(parent_xform);
    joint_X_c.append(child_xform);
    joint_name.append(name.value_or("joint {joint_count}"));
    joint_axis_start.append(joint_axis.width());
    joint_axis_dim.append(std::make_pair(linear_axes.size(), angular_axes.size()));
    joint_axis_total_count += linear_axes.size() + angular_axes.size();

    joint_linear_compliance.append(linear_compliance);
    joint_angular_compliance.append(angular_compliance);
    joint_enabled.append(enabled);

    for (const auto &dim : linear_axes) {
        _add_axis_dim(dim);
    }
    for (const auto &dim : angular_axes) {
        _add_axis_dim(dim);
    }

    size_t dof_count, coord_count;
    switch (type) {
        case JointType::JOINT_PRISMATIC:
        case JointType::JOINT_REVOLUTE:
            dof_count = 1;
            coord_count = 1;
            break;
        case JointType::JOINT_BALL:
            dof_count = 3;
            coord_count = 4;
            break;
        case JointType::JOINT_FREE:
        case JointType::JOINT_DISTANCE:
            dof_count = 6;
            coord_count = 7;
            break;
        case JointType::JOINT_FIXED:
            dof_count = 0;
            coord_count = 0;
            break;
        case JointType::JOINT_COMPOUND:
            dof_count = 3;
            coord_count = 3;
            break;
        case JointType::JOINT_UNIVERSAL:
            dof_count = 2;
            coord_count = 2;
            break;
        case JointType::JOINT_D6:
            dof_count = linear_axes.size() + angular_axes.size();
            coord_count = dof_count;
    }

    for (size_t i = 0; i < coord_count; i++) {
        joint_q.append(0.0);
    }
    for (size_t i = 0; i < dof_count; i++) {
        joint_qd.append(0.0);
        joint_act.append(0.0);
    }

    if (type == JointType::JOINT_FREE || type == JointType::JOINT_DISTANCE || type == JointType::JOINT_BALL)
        // ensure that a valid quaternion is used for the angular dofs
        *(joint_q.end() - 1) = 1.0;

    joint_q_start.append(joint_coord_count);
    joint_qd_start.append(joint_dof_count);

    joint_dof_count += dof_count;
    joint_coord_count += coord_count;

    if (collision_filter_parent && parent > -1)
        for (auto &child_shape : builder.body_shapes[child])
            for (auto &parent_shape : builder.body_shapes[parent])
                builder.shape_collision_filter_pairs.emplace(parent_shape, child_shape);

    return joint_count() - 1;
}

size_t JointModelBuilder::add_joint_revolute(int parent,
                                             int child,
                                             const TransformF &parent_xform,
                                             const TransformF &child_xform,
                                             const Vector3F &axis,
                                             float target,
                                             float target_ke,
                                             float target_kd,
                                             JointMode mode,
                                             float limit_lower,
                                             float limit_upper,
                                             float limit_ke,
                                             float limit_kd,
                                             float linear_compliance,
                                             float angular_compliance,
                                             std::optional<std::string_view> name,
                                             bool collision_filter_parent,
                                             bool enabled) {
    auto ax = JointAxis(axis);

    ax.limit_lower = limit_lower;
    ax.limit_upper = limit_upper;
    ax.target = target;
    ax.target_ke = target_ke;
    ax.target_kd = target_kd;
    ax.mode = mode;
    ax.limit_ke = limit_ke;
    ax.limit_kd = limit_kd;

    return add_joint(JointType::JOINT_REVOLUTE,
                     parent,
                     child,
                     {}, {ax}, name,
                     parent_xform,
                     child_xform,
                     linear_compliance,
                     angular_compliance,
                     collision_filter_parent,
                     enabled);
}

size_t JointModelBuilder::add_joint_prismatic(int parent,
                                              int child,
                                              const TransformF &parent_xform,
                                              const TransformF &child_xform,
                                              const Vector3F &axis,
                                              float target,
                                              float target_ke,
                                              float target_kd,
                                              JointMode mode,
                                              float limit_lower,
                                              float limit_upper,
                                              float limit_ke,
                                              float limit_kd,
                                              float linear_compliance,
                                              float angular_compliance,
                                              std::optional<std::string_view> name,
                                              bool collision_filter_parent,
                                              bool enabled) {
    auto ax = JointAxis(axis);

    ax.limit_lower = limit_lower;
    ax.limit_upper = limit_upper;
    ax.target = target;
    ax.target_ke = target_ke;
    ax.target_kd = target_kd;
    ax.mode = mode;
    ax.limit_ke = limit_ke;
    ax.limit_kd = limit_kd;
    return add_joint(JointType::JOINT_PRISMATIC,
                     parent,
                     child,
                     {ax}, {}, name,
                     parent_xform,
                     child_xform,
                     linear_compliance,
                     angular_compliance,
                     collision_filter_parent,
                     enabled);
}

size_t JointModelBuilder::add_joint_ball(int parent,
                                         int child,
                                         const TransformF &parent_xform,
                                         const TransformF &child_xform,
                                         float linear_compliance,
                                         float angular_compliance,
                                         std::optional<std::string_view> name,
                                         bool collision_filter_parent,
                                         bool enabled) {
    return add_joint(JointType::JOINT_BALL,
                     parent,
                     child,
                     {}, {}, name,
                     parent_xform,
                     child_xform,
                     linear_compliance,
                     angular_compliance,
                     collision_filter_parent,
                     enabled);
}

size_t JointModelBuilder::add_joint_fixed(int parent,
                                          int child,
                                          const TransformF &parent_xform,
                                          const TransformF &child_xform,
                                          float linear_compliance,
                                          float angular_compliance,
                                          std::optional<std::string_view> name,
                                          bool collision_filter_parent,
                                          bool enabled) {
    return add_joint(JointType::JOINT_FIXED,
                     parent,
                     child,
                     {}, {}, name,
                     parent_xform,
                     child_xform,
                     linear_compliance,
                     angular_compliance,
                     collision_filter_parent,
                     enabled);
}

size_t JointModelBuilder::add_joint_free(int child,
                                         const TransformF &parent_xform,
                                         const TransformF &child_xform,
                                         int parent,
                                         std::optional<std::string_view> name,
                                         bool collision_filter_parent,
                                         bool enabled) {
    return add_joint(JointType::JOINT_FREE,
                     parent,
                     child,
                     {}, {}, name,
                     parent_xform,
                     child_xform,
                     collision_filter_parent,
                     enabled);
}

size_t JointModelBuilder::add_joint_distance(int parent,
                                             int child,
                                             const TransformF &parent_xform,
                                             const TransformF &child_xform,
                                             float min_distance,
                                             float max_distance,
                                             float compliance,
                                             bool collision_filter_parent,
                                             bool enabled) {
    auto ax = JointAxis({1.0, 0.0, 0.0});
    ax.limit_lower = min_distance;
    ax.limit_upper = max_distance;

    return add_joint(JointType::JOINT_DISTANCE,
                     parent,
                     child,
                     {ax}, {}, std::nullopt,
                     parent_xform,
                     child_xform,
                     compliance,
                     0,
                     collision_filter_parent,
                     enabled);
}

size_t JointModelBuilder::add_joint_universal(int parent,
                                              int child,
                                              const JointAxis &axis_0,
                                              const JointAxis &axis_1,
                                              const TransformF &parent_xform,
                                              const TransformF &child_xform,
                                              float linear_compliance,
                                              float angular_compliance,
                                              std::optional<std::string_view> name,
                                              bool collision_filter_parent,
                                              bool enabled) {
    return add_joint(JointType::JOINT_UNIVERSAL,
                     parent,
                     child,
                     {}, {axis_0, axis_1}, name,
                     parent_xform,
                     child_xform,
                     linear_compliance,
                     angular_compliance,
                     collision_filter_parent,
                     enabled);
}

size_t JointModelBuilder::add_joint_compound(int parent,
                                             int child,
                                             const JointAxis &axis_0,
                                             const JointAxis &axis_1,
                                             const JointAxis &axis_2,
                                             const TransformF &parent_xform,
                                             const TransformF &child_xform,
                                             std::optional<std::string_view> name,
                                             bool collision_filter_parent,
                                             bool enabled) {
    return add_joint(JointType::JOINT_COMPOUND,
                     parent,
                     child,
                     {}, {axis_0, axis_1, axis_2}, name,
                     parent_xform,
                     child_xform,
                     0, 0,
                     collision_filter_parent,
                     enabled);
}

size_t JointModelBuilder::add_joint_d6(int parent,
                                       int child,
                                       std::initializer_list<JointAxis> linear_axes,
                                       std::initializer_list<JointAxis> angular_axes,
                                       std::optional<std::string_view> name,
                                       const TransformF &parent_xform,
                                       const TransformF &child_xform,
                                       float linear_compliance,
                                       float angular_compliance,
                                       bool collision_filter_parent,
                                       bool enabled) {
    return add_joint(JointType::JOINT_D6,
                     parent,
                     child,
                     linear_axes, angular_axes, name,
                     parent_xform,
                     child_xform,
                     linear_compliance,
                     angular_compliance,
                     collision_filter_parent,
                     enabled);
}

void JointModelBuilder::collapse_fixed_joints() {
    // todo
}

}// namespace vox