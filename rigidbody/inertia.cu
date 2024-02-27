//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "inertia.h"

namespace vox {
std::tuple<float, Vector3F, Matrix3x3F> compute_sphere_inertia(float density, float r) {
    float v = 4.f / 3.f * kPiF * r * r * r;
    float m = density * v;
    float Ia = 2.f / 5.f * m * r * r;

    Matrix3x3F I = Matrix3x3F({{Ia, 0.0, 0.0}, {0.0, Ia, 0.0}, {0.0, 0.0, Ia}});

    return {m, Vector3F(), I};
}

std::tuple<float, Vector3F, Matrix3x3F> compute_capsule_inertia(float density, float r, float h) {
    float ms = density * (4.f / 3.f) * kPiF * r * r * r;
    float mc = density * kPiF * r * r * h;

    // total mass
    float m = ms + mc;

    // adapted from ODE
    float Ia = mc * (0.25f * r * r + (1.f / 12.f) * h * h) + ms * (0.4f * r * r + 0.375f * r * h + 0.25f * h * h);
    float Ib = (mc * 0.5f + ms * 0.4f) * r * r;

    Matrix3x3F I = Matrix3x3F({{Ia, 0.0, 0.0}, {0.0, Ib, 0.0}, {0.0, 0.0, Ia}});

    return {m, Vector3F(), I};
}

std::tuple<float, Vector3F, Matrix3x3F> compute_cylinder_inertia(float density, float r, float h) {
    float m = density * kPiF * r * r * h;
    float Ia = 1.f / 12 * m * (3 * r * r + h * h);
    float Ib = 1.f / 2 * m * r * r;

    Matrix3x3F I = Matrix3x3F({{Ia, 0.0, 0.0}, {0.0, Ib, 0.0}, {0.0, 0.0, Ia}});

    return {m, Vector3F(), I};
}

std::tuple<float, Vector3F, Matrix3x3F> compute_cone_inertia(float density, float r, float h) {
    float m = density * kPiF * r * r * h / 3.f;

    float Ia = 1.f / 20 * m * (3 * r * r + 2 * h * h);
    float Ib = 3.f / 10 * m * r * r;

    Matrix3x3F I = Matrix3x3F({{Ia, 0.0, 0.0}, {0.0, Ib, 0.0}, {0.0, 0.0, Ia}});

    return {m, Vector3F(), I};
}

std::tuple<float, Vector3F, Matrix3x3F> compute_box_inertia(float density, float w, float h, float d) {
    float v = w * h * d;
    float m = density * v;

    float Ia = 1.f / 12.f * m * (h * h + d * d);
    float Ib = 1.f / 12.f * m * (w * w + d * d);
    float Ic = 1.f / 12.f * m * (w * w + h * h);

    Matrix3x3F I = Matrix3x3F({{Ia, 0.0, 0.0}, {0.0, Ib, 0.0}, {0.0, 0.0, Ic}});

    return {m, Vector3F(), I};
}

void compute_mesh_inertia(float density, const Tensor1<float> &vertices,
                          const Tensor1<float> &indices, bool is_solid,
                          const Tensor1<float> &thickness) {
    // todo
}

Matrix3x3F transform_inertia(float m, const Matrix3x3F &I, const Vector3F &p, const QuaternionF &q) {
    // todo
    return {};
}
}// namespace vox