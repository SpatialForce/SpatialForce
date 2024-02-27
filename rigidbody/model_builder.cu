//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "model_builder.h"

namespace vox {
ModelBuilder::ModelBuilder(const Vector3F &up_vector, float gravity)
    : up_vector{up_vector}, gravity{gravity},
      joint_builder{*this} {
}

size_t ModelBuilder::shape_count() {
    return shape_geo_type.width();
}

size_t ModelBuilder::body_count() {
    return body_q.width();
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

void ModelBuilder::add_builder() {}

size_t ModelBuilder::add_body(const TransformF &origin,
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