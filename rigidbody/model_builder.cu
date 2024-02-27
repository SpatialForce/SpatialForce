//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "model_builder.h"
#include "inertia.h"

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

size_t ModelBuilder::add_shape_plane(const Vector4F &plane,
                                     const std::optional<Vector3F> &pos,
                                     const std::optional<QuaternionF> &rot,
                                     float width,
                                     float length,
                                     int body,
                                     float ke,
                                     float kd,
                                     float kf,
                                     float mu,
                                     float restitution,
                                     float thickness,
                                     bool has_ground_collision) {
    Vector3F position;
    QuaternionF rotation;
    if (!pos.has_value() || !rot.has_value()) {
        // compute position and rotation from plane equation
        auto normal = Vector3F(plane.x, plane.y, plane.z);
        normal.normalize();
        position = plane[3] * normal;
        if (normal.distanceSquaredTo(Vector3F{0.0, 1.0, 0.0}) < kEpsilonF) {
            // no rotation necessary
            rotation = {0.0, 0.0, 0.0, 1.0};
        } else {
            auto c = normal.cross(Vector3F(0.0, 1.0, 0.0));
            auto angle = std::asin(c.norm());
            auto axis = c.normalized();
            rotation = QuaternionF(axis, angle);
        }
    } else {
        position = pos.value();
        rotation = rot.value();
    }
    auto scale = Vector3F(width, length, 0.0);

    return _add_shape(
        body,
        position,
        rotation,
        GeometryType::GEO_PLANE,
        scale,
        std::nullopt,
        0.0,
        ke,
        kd,
        kf,
        mu,
        restitution,
        thickness,
        true, -1, true,
        has_ground_collision);
}

size_t ModelBuilder::add_shape_sphere(int body,
                                      const Vector3F &pos,
                                      const QuaternionF &rot,
                                      float radius,
                                      float density,
                                      float ke,
                                      float kd,
                                      float kf,
                                      float mu,
                                      float restitution,
                                      bool is_solid,
                                      float thickness,
                                      bool has_ground_collision) {
    return _add_shape(body,
                      pos,
                      rot,
                      GeometryType::GEO_SPHERE,
                      Vector3F(radius, 0.0, 0.0),
                      std::nullopt,
                      density,
                      ke,
                      kd,
                      kf,
                      mu,
                      restitution,
                      thickness + radius,
                      is_solid, -1, true,
                      has_ground_collision);
}

size_t ModelBuilder::add_shape_box(int body,
                                   const Vector3F &pos,
                                   const QuaternionF &rot,
                                   float hx,
                                   float hy,
                                   float hz,
                                   float density,
                                   float ke,
                                   float kd,
                                   float kf,
                                   float mu,
                                   float restitution,
                                   bool is_solid,
                                   float thickness,
                                   bool has_ground_collision) {
    return _add_shape(body,
                      pos,
                      rot,
                      GeometryType::GEO_BOX,
                      Vector3F(hx, hy, hz),
                      std::nullopt,
                      density,
                      ke,
                      kd,
                      kf,
                      mu,
                      restitution,
                      thickness,
                      is_solid, -1, true,
                      has_ground_collision);
}

size_t ModelBuilder::add_shape_capsule(int body,
                                       const Vector3F &pos,
                                       const QuaternionF &rot,
                                       float radius,
                                       float half_height,
                                       UpAxis axis,
                                       float density,
                                       float ke,
                                       float kd,
                                       float kf,
                                       float mu,
                                       float restitution,
                                       bool is_solid,
                                       float thickness,
                                       bool has_ground_collision) {
    auto q = rot;
    float sqh = std::sqrt(0.5f);
    if (axis == UpAxis::X) {
        q *= QuaternionF(0.0, 0.0, -sqh, sqh);
    } else if (axis == UpAxis::Z) {
        q *= QuaternionF(sqh, 0.0, 0.0, sqh);
    }

    return _add_shape(body,
                      pos,
                      q,
                      GeometryType::GEO_CAPSULE,
                      Vector3F(radius, half_height, 0.0),
                      std::nullopt,
                      density,
                      ke,
                      kd,
                      kf,
                      mu,
                      restitution,
                      thickness + radius,
                      is_solid, -1, true,
                      has_ground_collision);
}

size_t ModelBuilder::add_shape_cylinder(int body,
                                        const Vector3F &pos,
                                        const QuaternionF &rot,
                                        float radius,
                                        float half_height,
                                        UpAxis axis,
                                        float density,
                                        float ke,
                                        float kd,
                                        float kf,
                                        float mu,
                                        float restitution,
                                        bool is_solid,
                                        float thickness,
                                        bool has_ground_collision) {
    auto q = rot;
    float sqh = std::sqrt(0.5f);
    if (axis == UpAxis::X) {
        q *= QuaternionF(0.0, 0.0, -sqh, sqh);
    } else if (axis == UpAxis::Z) {
        q *= QuaternionF(sqh, 0.0, 0.0, sqh);
    }

    return _add_shape(
        body,
        pos,
        q,
        GeometryType::GEO_CYLINDER,
        Vector3F(radius, half_height, 0.0),
        std::nullopt,
        density,
        ke,
        kd,
        kf,
        mu,
        restitution,
        thickness,
        is_solid, -1, true,
        has_ground_collision);
}

size_t ModelBuilder::add_shape_cone(int body,
                                    const Vector3F &pos,
                                    const QuaternionF &rot,
                                    float radius,
                                    float half_height,
                                    UpAxis axis,
                                    float density,
                                    float ke,
                                    float kd,
                                    float kf,
                                    float mu,
                                    float restitution,
                                    bool is_solid,
                                    float thickness,
                                    bool has_ground_collision) {
    auto q = rot;
    float sqh = std::sqrt(0.5f);
    if (axis == UpAxis::X) {
        q *= QuaternionF(0.0, 0.0, -sqh, sqh);
    } else if (axis == UpAxis::Z) {
        q *= QuaternionF(sqh, 0.0, 0.0, sqh);
    }

    return _add_shape(
        body,
        pos,
        q,
        GeometryType::GEO_CONE,
        Vector3F(radius, half_height, 0.0),
        std::nullopt,
        density,
        ke,
        kd,
        kf,
        mu,
        restitution,
        thickness,
        is_solid, -1, true,
        has_ground_collision);
}

size_t ModelBuilder::add_shape_mesh(int body,
                                    const Vector3F &pos,
                                    const QuaternionF &rot,
                                    std::optional<int> mesh,
                                    const Vector3F &scale,
                                    float density,
                                    float ke,
                                    float kd,
                                    float kf,
                                    float mu,
                                    float restitution,
                                    bool is_solid,
                                    float thickness,
                                    bool has_ground_collision) {
    return _add_shape(body,
                      pos,
                      rot,
                      GeometryType::GEO_MESH,
                      scale,
                      mesh,
                      density,
                      ke,
                      kd,
                      kf,
                      mu,
                      restitution,
                      thickness,
                      is_solid, -1, true,
                      has_ground_collision);
}

size_t ModelBuilder::add_shape_sdf(int body,
                                   const Vector3F &pos,
                                   const QuaternionF &rot,
                                   std::optional<int> sdf,
                                   const Vector3F &scale,
                                   float density,
                                   float ke,
                                   float kd,
                                   float kf,
                                   float mu,
                                   float restitution,
                                   bool is_solid,
                                   float thickness,
                                   bool has_ground_collision) {
    return _add_shape(body,
                      pos,
                      rot,
                      GeometryType::GEO_SDF,
                      scale,
                      sdf,
                      density,
                      ke,
                      kd,
                      kf,
                      mu,
                      restitution,
                      thickness,
                      is_solid, -1, true,
                      has_ground_collision);
}

float ModelBuilder::_shape_radius(GeometryType type, const Vector3F &scale, std::optional<int> src) {
    if (type == GeometryType::GEO_SPHERE) {
        return scale[0];
    } else if (type == GeometryType::GEO_BOX) {
        return scale.norm();
    } else if (type == GeometryType::GEO_CAPSULE || type == GeometryType::GEO_CYLINDER || type == GeometryType::GEO_CONE) {
        return scale[0] + scale[1];
    } else if (type == GeometryType::GEO_MESH) {
        // todo
        // auto vmax = np.max(np.abs(src.vertices), axis = 0) * np.max(scale);
        // return np.linalg.norm(vmax);
    } else if (type == GeometryType::GEO_PLANE) {
        if (scale[0] > 0.0 && scale[1] > 0.0) {
            // finite plane
            return scale.norm();
        } else {
            return 1.0e6;
        }
    } else {
        return 10.0;
    }
}

std::tuple<float, Vector3F, Matrix3x3F> compute_shape_mass(GeometryType type, const Vector3F &scale,
                                                           std::optional<int> src, float density,
                                                           bool is_solid, float thickness) {
    if (density == 0.0 || type == GeometryType::GEO_PLANE)// zero density means fixed
        return {0.0, Vector3F(), Matrix3x3F()};

    if (type == GeometryType::GEO_SPHERE) {
        auto solid = compute_sphere_inertia(density, scale[0]);
        if (is_solid) {
            return solid;
        } else {
            auto hollow = compute_sphere_inertia(density, scale[0] - thickness);
            return {std::get<0>(solid) - std::get<0>(hollow), std::get<1>(solid), std::get<2>(solid) - std::get<2>(hollow)};
        }
    } else if (type == GeometryType::GEO_BOX) {
        Vector3F scale2 = scale * 2.f;
        auto w = scale2.x;
        auto h = scale2.y;
        auto d = scale2.z;
        auto solid = compute_box_inertia(density, w, h, d);
        if (is_solid) {
            return solid;
        } else {
            auto hollow = compute_box_inertia(density, w - thickness, h - thickness, d - thickness);
            return {std::get<0>(solid) - std::get<0>(hollow), std::get<1>(solid), std::get<2>(solid) - std::get<2>(hollow)};
        }
    } else if (type == GeometryType::GEO_CAPSULE) {
        auto r = scale[0];
        auto h = scale[1] * 2.f;
        auto solid = compute_capsule_inertia(density, r, h);
        if (is_solid) {
            return solid;
        } else {
            auto hollow = compute_capsule_inertia(density, r - thickness, h - 2.f * thickness);
            return {std::get<0>(solid) - std::get<0>(hollow), std::get<1>(solid), std::get<2>(solid) - std::get<2>(hollow)};
        }
    } else if (type == GeometryType::GEO_CYLINDER) {
        auto r = scale[0];
        auto h = scale[1] * 2.f;
        auto solid = compute_cylinder_inertia(density, r, h);
        if (is_solid) {
            return solid;
        } else {
            auto hollow = compute_cylinder_inertia(density, r - thickness, h - 2.f * thickness);
            return {std::get<0>(solid) - std::get<0>(hollow), std::get<1>(solid), std::get<2>(solid) - std::get<2>(hollow)};
        }
    } else if (type == GeometryType::GEO_CONE) {
        auto r = scale[0];
        auto h = scale[1] * 2.f;
        auto solid = compute_cone_inertia(density, r, h);
        if (is_solid) {
            return solid;
        } else {
            auto hollow = compute_cone_inertia(density, r - thickness, h - 2.f * thickness);
            return {std::get<0>(solid) - std::get<0>(hollow), std::get<1>(solid), std::get<2>(solid) - std::get<2>(hollow)};
        }
    } else if (type == GeometryType::GEO_MESH || type == GeometryType::GEO_SDF) {
        // todo
        //        if src.has_inertia and src.mass > 0.0 and src.is_solid == is_solid:
        //            m, c, I = src.mass, src.com, src.I
        //
        //            sx, sy, sz = scale
        //
        //            mass_ratio = sx * sy * sz * density
        //            m_new = m * mass_ratio
        //
        //            c_new = wp.cw_mul(c, scale)
        //
        //            Ixx = I[0, 0] * (sy**2 + sz**2) / 2 * mass_ratio
        //            Iyy = I[1, 1] * (sx**2 + sz**2) / 2 * mass_ratio
        //            Izz = I[2, 2] * (sx**2 + sy**2) / 2 * mass_ratio
        //            Ixy = I[0, 1] * sx * sy * mass_ratio
        //            Ixz = I[0, 2] * sx * sz * mass_ratio
        //            Iyz = I[1, 2] * sy * sz * mass_ratio
        //
        //            I_new = wp.mat33([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
        //
        //            return m_new, c_new, I_new
    } else if (type == GeometryType::GEO_MESH) {
        // todo
        //            # fall back to computing inertia from mesh geometry
        //            vertices = np.array(src.vertices) * np.array(scale)
        //            m, c, I, vol = compute_mesh_inertia(density, vertices, src.indices, is_solid, thickness)
        //            return m, c, I
    }
}

size_t ModelBuilder::_add_shape(int body,
                                const Vector3F &pos,
                                const QuaternionF &rot,
                                GeometryType type,
                                const Vector3F &scale,
                                std::optional<int> src,
                                float density,
                                float ke,
                                float kd,
                                float kf,
                                float mu,
                                float restitution,
                                float thickness,
                                bool is_solid,
                                int collision_group,
                                bool collision_filter_parent,
                                bool has_ground_collision) {
    shape_body.append(body);
    auto shape = shape_count();
    if (auto iter = body_shapes.find(body); iter != body_shapes.end()) {
        // no contacts between shapes of the same body
        for (auto &same_body_shape : iter->second)
            shape_collision_filter_pairs.emplace(same_body_shape, shape);
        body_shapes[body].append(shape);
    } else {
        body_shapes[body] = {shape};
    }
    shape_transform.append(TransformF(pos, rot));
    shape_geo_type.append(type);
    shape_geo_scale.append((scale[0], scale[1], scale[2]));
    shape_geo_src.append(src);// todo
    shape_geo_thickness.append(thickness);
    shape_geo_is_solid.append(is_solid);
    shape_material_ke.append(ke);
    shape_material_kd.append(kd);
    shape_material_kf.append(kf);
    shape_material_mu.append(mu);
    shape_material_restitution.append(restitution);
    shape_collision_group.append(collision_group);
    if (auto iter = shape_collision_group_map.find(collision_group); iter == shape_collision_group_map.end())
        shape_collision_group_map.emplace(collision_group, Tensor1<size_t>{});
    last_collision_group = std::max(last_collision_group, collision_group);
    shape_collision_group_map[collision_group].append(shape);
    shape_collision_radius.append(_shape_radius(type, scale, src));
    if (collision_filter_parent && body > -1 && joint_builder.joint_parents.contains(body))
        for (auto &parent_body : joint_builder.joint_parents[body])
            if (parent_body > -1)
                for (auto &parent_shape : body_shapes[parent_body])
                    shape_collision_filter_pairs.emplace(parent_shape, shape);
    if (body == -1)
        has_ground_collision = false;
    shape_ground_collision.append(has_ground_collision);

    float m;
    Vector3F c;
    Matrix3x3F I;
    std::tie(m, c, I) = compute_shape_mass(type, scale, src, density, is_solid, thickness);

    _update_body_mass(body, m, I, pos + c, rot);
    return shape;
}

void ModelBuilder::_update_body_mass(int i, float m,
                                     const Matrix3x3F &I,
                                     const Vector3F &p,
                                     const QuaternionF &q) {
    if (i == -1)
        return;

    // find new COM
    float new_mass = body_mass[i] + m;

    if (new_mass == 0.0)// no mass
        return;

    Vector3F new_com = (body_com[i] * body_mass[i] + p * m) / new_mass;

    // shift inertia to new COM
    auto com_offset = new_com - body_com[i];
    auto shape_offset = new_com - p;

    auto new_inertia = transform_inertia(body_mass[i], body_inertia[i], com_offset, QuaternionF()) +
                       transform_inertia(m, I, shape_offset, q);

    body_mass[i] = new_mass;
    body_inertia[i] = new_inertia;
    body_com[i] = new_com;

    if (new_mass > 0.0) {
        body_inv_mass[i] = 1.f / new_mass;
    } else {
        body_inv_mass[i] = 0.f;
    }
    if (new_inertia.determinant() < kEpsilonF) {
        body_inv_inertia[i] = new_inertia.inverse();
    } else {
        body_inv_inertia[i] = new_inertia;
    }
}

void ModelBuilder::set_ground_plane(const Vector3F &normal,
                                    float offset,
                                    float ke,
                                    float kd,
                                    float kf,
                                    float mu,
                                    float restitution) {
    _ground_params = {
        .plane = Vector4F(normal.x, normal.y, normal.z, offset),
        .width = 0.0,
        .length = 0.0,
        .ke = ke,
        .kd = kd,
        .kf = kf,
        .mu = mu,
        .restitution = restitution};
}

void ModelBuilder::_create_ground_plane() {
    auto ground_id = add_shape_plane(_ground_params.plane, std::nullopt, std::nullopt,
                                     _ground_params.width, _ground_params.length, -1,
                                     _ground_params.ke, _ground_params.kd, _ground_params.kf, _ground_params.mu,
                                     _ground_params.restitution);
    _ground_created = true;
    // disable ground collisions as they will be treated separately
    for (size_t i = 0; i < shape_count() - 1; i++) {
        shape_collision_filter_pairs.emplace(i, ground_id);
    }
}

void ModelBuilder::finalize(uint32_t index) {}
}// namespace vox