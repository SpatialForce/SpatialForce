//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "particles.h"

namespace vox {
size_t ParticleModelBuilder::particle_count() {
    return particle_q.width();
}

size_t ParticleModelBuilder::tri_count() {
    return tri_poses.width();
}

size_t ParticleModelBuilder::tet_count() {
    return tet_poses.width();
}

size_t ParticleModelBuilder::edge_count() {
    return edge_rest_angle.width();
}

size_t ParticleModelBuilder::spring_count() {
    return spring_rest_length.width();
}

size_t ParticleModelBuilder::add_particle(const Vector3F &pos,
                                          const Vector3F &vel,
                                          float mass,
                                          float radius,
                                          PARTICLE_FLAG flags) {
    particle_q.append(pos);
    particle_qd.append(vel);
    particle_mass.append(mass);
    particle_radius.append(radius);
    particle_flags.append(int(flags));

    return particle_q.width() - 1;
}

void ParticleModelBuilder::add_spring(int i, int j, float ke, float kd, float control) {
    spring_indices.append(i);
    spring_indices.append(j);
    spring_stiffness.append(ke);
    spring_damping.append(kd);
    spring_control.append(control);

    // compute rest length
    auto p = particle_q[i];
    auto q = particle_q[j];

    auto delta = p - q;
    auto l = delta.length();

    spring_rest_length.append(l);
}

float ParticleModelBuilder::add_triangle(int i,
                                         int j,
                                         int k,
                                         float tri_ke,
                                         float tri_ka,
                                         float tri_kd,
                                         float tri_drag,
                                         float tri_lift) {
    // compute basis for 2D rest pose
    Vector3F p = particle_q[i];
    Vector3F q = particle_q[j];
    Vector3F r = particle_q[k];

    Vector3F qp = q - p;
    Vector3F rp = r - p;

    // construct basis aligned with the triangle
    Vector3F n = qp.cross(rp).normalized();
    Vector3F e1 = qp.normalized();
    Vector3F e2 = n.cross(e1).normalized();

    auto R = Matrix<float, 3, 2>({{e1.x, e1.y, e1.z}, {e2.x, e2.y, e2.z}});
    auto M = Matrix<float, 3, 2>({{qp.x, qp.y, qp.z}, {rp.x, rp.y, rp.z}});
    Matrix3x3F D = R * M.transposed();

    float area = D.determinant() / 2.f;

    if (area <= 0.0) {
        // print("inverted or degenerate triangle element");
        return 0.0;
    } else {
        auto inv_D = D.inverse();

        tri_indices.append({i, j, k});
        tri_poses.append(inv_D);
        tri_activations.append(0.0);
        tri_materials.append({tri_ke, tri_ka, tri_kd, tri_drag, tri_lift});
        return area;
    }
}

std::vector<float> ParticleModelBuilder::add_triangles(std::initializer_list<int> i,
                                                       std::initializer_list<int> j,
                                                       std::initializer_list<int> k,
                                                       std::initializer_list<float> tri_ke,
                                                       std::initializer_list<float> tri_ka,
                                                       std::initializer_list<float> tri_kd,
                                                       std::initializer_list<float> tri_drag,
                                                       std::initializer_list<float> tri_lift) {
    return {};
}

float ParticleModelBuilder::add_tetrahedron(int i, int j, int k, int l, float k_mu, float k_lambda, float k_damp) {
    return 0;
}

void ParticleModelBuilder::add_edge(int i,
                                    int j,
                                    int k,
                                    int l,
                                    std::optional<float> rest,
                                    float edge_ke,
                                    float edge_kd) {
}

void ParticleModelBuilder::add_edges(const std::vector<int> &i,
                                     const std::vector<int> &j,
                                     const std::vector<int> &k,
                                     const std::vector<int> &l,
                                     const std::optional<std::vector<float>> &rest,
                                     const std::optional<std::vector<float>> &edge_ke,
                                     const std::optional<std::vector<float>> &edge_kd) {
}

void ParticleModelBuilder::add_cloth_grid(const Vector3F &pos,
                                          const QuaternionF &rot,
                                          const Vector3F &vel,
                                          int dim_x,
                                          int dim_y,
                                          float cell_x,
                                          float cell_y,
                                          float mass,
                                          bool reverse_winding,
                                          bool fix_left,
                                          bool fix_right,
                                          bool fix_top,
                                          bool fix_bottom,
                                          float tri_ke,
                                          float tri_ka,
                                          float tri_kd,
                                          float tri_drag,
                                          float tri_lift,
                                          float edge_ke,
                                          float edge_kd,
                                          bool add_springs,
                                          float spring_ke,
                                          float spring_kd) {
}

void ParticleModelBuilder::add_cloth_mesh(const Vector3F &pos,
                                          const QuaternionF &rot,
                                          float scale,
                                          const Vector3F &vel,
                                          const std::vector<Vector3F> &vertices,
                                          const std::vector<int> &indices,
                                          float density,
                                          float tri_ke,
                                          float tri_ka,
                                          float tri_kd,
                                          float tri_drag,
                                          float tri_lift,
                                          float edge_ke,
                                          float edge_kd,
                                          bool add_springs,
                                          float spring_ke,
                                          float spring_kd) {
}

void ParticleModelBuilder::add_particle_grid(const Vector3F &pos,
                                             const QuaternionF &rot,
                                             const Vector3F &vel,
                                             int dim_x,
                                             int dim_y,
                                             int dim_z,
                                             float cell_x,
                                             float cell_y,
                                             float cell_z,
                                             float mass,
                                             float jitter,
                                             float radius_mean,
                                             float radius_std) {
}

void ParticleModelBuilder::add_soft_grid(const Vector3F &pos,
                                         const QuaternionF &rot,
                                         const Vector3F &vel,
                                         int dim_x,
                                         int dim_y,
                                         int dim_z,
                                         float cell_x,
                                         float cell_y,
                                         float cell_z,
                                         float density,
                                         float k_mu,
                                         float k_lambda,
                                         float k_damp,
                                         bool fix_left,
                                         bool fix_right,
                                         bool fix_top,
                                         bool fix_bottom,
                                         float tri_ke,
                                         float tri_ka,
                                         float tri_kd,
                                         float tri_drag,
                                         float tri_lift) {
}

void ParticleModelBuilder::add_soft_mesh(const Vector3F &pos,
                                         const QuaternionF &rot,
                                         float scale,
                                         const Vector3F &vel,
                                         const std::vector<Vector3F> &vertices,
                                         const std::vector<int> &indices,
                                         float density,
                                         float k_mu,
                                         float k_lambda,
                                         float k_damp,
                                         float tri_ke,
                                         float tri_ka,
                                         float tri_kd,
                                         float tri_drag,
                                         float tri_lift) {
}
}// namespace vox