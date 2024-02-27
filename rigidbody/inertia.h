//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/quaternion.h"
#include "tensor/tensor.h"

namespace vox {
/// Helper to compute mass and inertia of a solid sphere
/// \param density The sphere density
/// \param r The sphere radius
/// \return A tuple of (mass, inertia) with inertia specified around the origin
std::tuple<float, Vector3F, Matrix3x3F> compute_sphere_inertia(float density, float r);

/// Helper to compute mass and inertia of a solid capsule extending along the y-axis
/// \param density The capsule density
/// \param r The capsule radius
/// \param h The capsule height (full height of the interior cylinder)
/// \return A tuple of (mass, inertia) with inertia specified around the origin
std::tuple<float, Vector3F, Matrix3x3F> compute_capsule_inertia(float density, float r, float h);

/// Helper to compute mass and inertia of a solid cylinder extending along the y-axis
/// \param density The cylinder density
/// \param r The cylinder radius
/// \param h The cylinder height (extent along the y-axis)
/// \return A tuple of (mass, inertia) with inertia specified around the origin
std::tuple<float, Vector3F, Matrix3x3F> compute_cylinder_inertia(float density, float r, float h);

/// Helper to compute mass and inertia of a solid cone extending along the y-axis
/// \param density The cone density
/// \param r The cone radius
/// \param h The cone height (extent along the y-axis)
/// \return A tuple of (mass, inertia) with inertia specified around the origin
std::tuple<float, Vector3F, Matrix3x3F> compute_cone_inertia(float density, float r, float h);

/// Helper to compute mass and inertia of a solid box
/// \param density The box density
/// \param w The box width along the x-axis
/// \param h The box height along the y-axis
/// \param d The box depth along the z-axis
/// \return A tuple of (mass, inertia) with inertia specified around the origin
std::tuple<float, Vector3F, Matrix3x3F> compute_box_inertia(float density, float w, float h, float d);

void compute_mesh_inertia(float density, const Tensor1<float> &vertices,
                          const Tensor1<float> &indices, bool is_solid = true,
                          const Tensor1<float> &thickness = {0.001});

Matrix3x3F transform_inertia(float m, const Matrix3x3F &I, const Vector3F &p, const QuaternionF &q);
}// namespace vox