//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "model_builder.h"

namespace vox {
ModelBuilder::ModelBuilder(const Vector3F &up_vector, float gravity)
    : up_vector{up_vector}, gravity{gravity} {
}

size_t ModelBuilder::shape_count() {
    return shape_geo_type.width();
}

size_t ModelBuilder::body_count() {
    return body_q.width();
}

size_t ModelBuilder::joint_count() {
    return joint_type.width();
}

size_t ModelBuilder::joint_axis_count() {
    return joint_axis.width();
}

size_t ModelBuilder::particle_count() {
    return particle_q.width();
}

size_t ModelBuilder::tri_count() {
    return tri_poses.width();
}

size_t ModelBuilder::tet_count() {
    return tet_poses.width();
}

size_t ModelBuilder::edge_count() {
    return edge_rest_angle.width();
}

size_t ModelBuilder::spring_count() {
    return spring_rest_length.width();
}

size_t ModelBuilder::muscle_count() {
    return muscle_start.width();
}

size_t ModelBuilder::articulation_count() {
    return articulation_start.width();
}

void ModelBuilder::add_articulation() {
    articulation_start.append(joint_count());
}

void ModelBuilder::add_builder() {}

size_t ModelBuilder::add_body(const Transform3 &origin,
                              float armature,
                              const Vector3F &com,
                              const Matrix3x3F &I_m,
                              float m,
                              std::optional<std::string_view> name) {
    auto body_id = body_mass.width();

    // body data
    auto inertia = I_m + Matrix3x3F::makeConstant(armature);
    body_inertia.append(inertia);
    body_mass.append(m);
    body_com.append(com);

    if (m > 0.0) {
        body_inv_mass.append(1.f / m);
    } else {
        body_inv_mass.append(0.0);
    }

    if (inertia.determinant() < kEpsilonF) {
        body_inv_inertia.append(inertia);
    } else {
        body_inv_inertia.append(inertia.inverse());
    }
    body_q.append(origin);
    body_qd.append(SpatialVectorF());

    if (name.has_value()) {
        body_name.append(name.value());
    } else {
        body_name.append("body {body_id}");
    }
    body_shapes[body_id] = {};
    return body_id;
}

void ModelBuilder::_add_axis_dim(const JointAxis &dim) {
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

size_t ModelBuilder::add_joint(JointType type,
                               int parent,
                               int child,
                               std::initializer_list<JointAxis> linear_axes,
                               std::initializer_list<JointAxis> angular_axes,
                               std::optional<std::string_view> name,
                               const Transform3 &parent_xform,
                               const Transform3 &child_xform,
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
        for (auto &child_shape : body_shapes[child])
            for (auto &parent_shape : body_shapes[parent])
                shape_collision_filter_pairs.emplace(parent_shape, child_shape);

    return joint_count() - 1;
}

void ModelBuilder::add_joint_revolute() {}

void ModelBuilder::add_joint_prismatic() {}

void ModelBuilder::add_joint_ball() {}

void ModelBuilder::add_joint_fixed() {}

void ModelBuilder::add_joint_free() {}

void ModelBuilder::add_joint_distance() {}

void ModelBuilder::add_joint_universal() {}

void ModelBuilder::add_joint_compound() {}

void ModelBuilder::add_joint_d6() {}

void ModelBuilder::collapse_fixed_joints() {}

void ModelBuilder::add_muscle() {}

void ModelBuilder::add_shape_plane() {}

void ModelBuilder::add_shape_sphere() {}

void ModelBuilder::add_shape_box() {}

void ModelBuilder::add_shape_capsule() {}

void ModelBuilder::add_shape_cylinder() {}

void ModelBuilder::add_shape_cone() {}

void ModelBuilder::add_shape_mesh() {}

void ModelBuilder::add_shape_sdf() {}

void ModelBuilder::_shape_radius() {}

void ModelBuilder::_add_shape() {}

void ModelBuilder::add_particle() {}

void ModelBuilder::add_spring() {}

void ModelBuilder::add_triangle() {}

void ModelBuilder::add_triangles() {}

void ModelBuilder::add_tetrahedron() {}

void ModelBuilder::add_edge() {}

void ModelBuilder::add_edges() {}

void ModelBuilder::add_cloth_grid() {}

void ModelBuilder::add_cloth_mesh() {}

void ModelBuilder::add_particle_grid() {}

void ModelBuilder::add_soft_grid() {}

void ModelBuilder::add_soft_mesh() {}

void ModelBuilder::_update_body_mass() {}

void ModelBuilder::set_ground_plane() {}

void ModelBuilder::_create_ground_plane() {}

void ModelBuilder::finalize(uint32_t index) {}
}// namespace vox