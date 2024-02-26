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

void ModelBuilder::add_body() {}

void ModelBuilder::add_joint() {}

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