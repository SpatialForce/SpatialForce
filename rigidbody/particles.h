//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "tensor/tensor.h"
#include "math/matrix.h"
#include "math/quaternion.h"
#include <optional>

namespace vox {
enum class PARTICLE_FLAG : int {
    ACTIVE = 1 << 0
};

class ParticleModelBuilder {
public:
    // particle settings
    static constexpr float default_particle_radius = 0.1;

    // triangle soft mesh settings
    static constexpr float default_tri_ke = 100.0;
    static constexpr float default_tri_ka = 100.0;
    static constexpr float default_tri_kd = 10.0;
    static constexpr float default_tri_drag = 0.0;
    static constexpr float default_tri_lift = 0.0;

    // distance constraint properties
    static constexpr float default_spring_ke = 100.0;
    static constexpr float default_spring_kd = 0.0;

    // edge bending properties
    static constexpr float default_edge_ke = 100.0;
    static constexpr float default_edge_kd = 0.0;

    Tensor1<Vector3F> particle_q;
    Tensor1<Vector3F> particle_qd;
    Tensor1<float> particle_mass;
    Tensor1<float> particle_radius;
    Tensor1<int> particle_flags;
    float particle_max_velocity = 1e5;

    // springs
    Tensor1<int> spring_indices;
    Tensor1<float> spring_rest_length;
    Tensor1<float> spring_stiffness;
    Tensor1<float> spring_damping;
    Tensor1<float> spring_control;

    // triangles
    Tensor1<std::tuple<int, int, int>> tri_indices;
    Tensor1<Matrix3x3F> tri_poses;
    Tensor1<float> tri_activations;
    struct TriMaterial {
        float tri_ke, tri_ka, tri_kd, tri_drag, tri_lift;
    };
    Tensor1<TriMaterial> tri_materials;

    // edges (bending)
    Tensor1<float> edge_indices;
    Tensor1<float> edge_rest_angle;
    Tensor1<float> edge_bending_properties;

    // tetrahedra
    Tensor1<float> tet_indices;
    Tensor1<float> tet_poses;
    Tensor1<float> tet_activations;
    Tensor1<float> tet_materials;

    size_t particle_count();

    size_t tri_count();

    size_t tet_count();

    size_t edge_count();

    size_t spring_count();

    /// Adds a single particle to the model
    /// \param pos The initial position of the particle
    /// \param vel The initial velocity of the particle
    /// \param mass The mass of the particle
    /// \param radius  The radius of the particle used in collision handling. If None, the radius is set to the default value (default_particle_radius).
    /// \param flags The flags that control the dynamical behavior of the particle, see PARTICLE_FLAG_* constants
    /// \return The index of the particle in the system
    /// \remark Set the mass equal to zero to create a 'kinematic' particle that does is not subject to dynamics.
    size_t add_particle(const Vector3F &pos, const Vector3F &vel, float mass,
                        float radius = default_particle_radius,
                        PARTICLE_FLAG flags = PARTICLE_FLAG::ACTIVE);

    /// Adds a spring between two particles in the system
    /// \param i The index of the first particle
    /// \param j The index of the second particle
    /// \param ke The elastic stiffness of the spring
    /// \param kd The damping stiffness of the spring
    /// \param control The actuation level of the spring
    /// \remark The spring is created with a rest-length based on the distance
    ///            between the particles in their initial configuration.
    void add_spring(int i, int j, float ke, float kd, float control);

    /// Adds a triangular FEM element between three particles in the system.
    //
    //        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
    //        Parameters specified on the model. See model.tri_ke, model.tri_kd.
    /// \param i The index of the first particle
    /// \param j The index of the second particle
    /// \param k The index of the third particle
    /// \param tri_ke
    /// \param tri_ka
    /// \param tri_kd
    /// \param tri_drag
    /// \param tri_lift
    /// \return The area of the triangle
    /// \remark The triangle is created with a rest-length based on the distance
    ///            between the particles in their initial configuration.
    float add_triangle(int i,
                       int j,
                       int k,
                       float tri_ke = default_tri_ke,
                       float tri_ka = default_tri_ka,
                       float tri_kd = default_tri_kd,
                       float tri_drag = default_tri_drag,
                       float tri_lift = default_tri_lift);

    /// Adds triangular FEM elements between groups of three particles in the system.
    //
    //        Triangles are modeled as viscoelastic elements with elastic stiffness and damping
    //        Parameters specified on the model. See model.tri_ke, model.tri_kd.
    /// \param i The indices of the first particle
    /// \param j The indices of the second particle
    /// \param k The indices of the third particle
    /// \param tri_ke
    /// \param tri_ka
    /// \param tri_kd
    /// \param tri_drag
    /// \param tri_lift
    /// \return The areas of the triangles
    /// \remark A triangle is created with a rest-length based on the distance
    ///            between the particles in their initial configuration.
    std::vector<float> add_triangles(std::initializer_list<int> i,
                                     std::initializer_list<int> j,
                                     std::initializer_list<int> k,
                                     std::initializer_list<float> tri_ke = {},
                                     std::initializer_list<float> tri_ka = {},
                                     std::initializer_list<float> tri_kd = {},
                                     std::initializer_list<float> tri_drag = {},
                                     std::initializer_list<float> tri_lift = {});

    /// Adds a tetrahedral FEM element between four particles in the system.
    //
    //        Tetrahedra are modeled as viscoelastic elements with a NeoHookean energy
    //        density based on [Smith et al. 2018].
    /// \param i The index of the first particle
    /// \param j The index of the second particle
    /// \param k The index of the third particle
    /// \param l The index of the fourth particle
    /// \param k_mu The first elastic Lame parameter
    /// \param k_lambda The second elastic Lame parameter
    /// \param k_damp The element's damping stiffness
    /// \return The volume of the tetrahedron
    /// \remark The tetrahedron is created with a rest-pose based on the particle's initial configuration
    float add_tetrahedron(int i, int j, int k, int l, float k_mu = 1.0e3, float k_lambda = 1.0e3, float k_damp = 0.0);

    /// Adds a bending edge element between four particles in the system.
    ///
    ///        Bending elements are designed to be between two connected triangles. Then
    ///        bending energy is based of [Bridson et al. 2002]. Bending stiffness is controlled
    ///        by the `model.tri_kb` parameter.
    /// \param i The index of the first particle
    /// \param j The index of the second particle
    /// \param k The index of the third particle
    /// \param l The index of the fourth particle
    /// \param rest The rest angle across the edge in radians, if not specified it will be computed
    /// \param edge_ke
    /// \param edge_kd
    /// \remark The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
    ///            vertices indexed by 'i' and 'j'. This defines two connected triangles with counter clockwise
    ///            winding: (i, k, l), (j, l, k).
    void add_edge(int i,
                  int j,
                  int k,
                  int l,
                  std::optional<float> rest = std::nullopt,
                  float edge_ke = default_edge_ke,
                  float edge_kd = default_edge_kd);

    /// Adds bending edge elements between groups of four particles in the system.
    ///
    ///        Bending elements are designed to be between two connected triangles. Then
    ///        bending energy is based of [Bridson et al. 2002]. Bending stiffness is controlled
    ///        by the `model.tri_kb` parameter.
    /// \param i The indices of the first particle
    /// \param j The indices of the second particle
    /// \param k The indices of the third particle
    /// \param l The indices of the fourth particle
    /// \param rest The rest angles across the edges in radians, if not specified they will be computed
    /// \param edge_ke
    /// \param edge_kd
    /// \remark The edge lies between the particles indexed by 'k' and 'l' parameters with the opposing
    ///            vertices indexed by 'i' and 'j'. This defines two connected triangles with counter clockwise
    ///            winding: (i, k, l), (j, l, k).
    void add_edges(const std::vector<int> &i,
                   const std::vector<int> &j,
                   const std::vector<int> &k,
                   const std::vector<int> &l,
                   const std::optional<std::vector<float>> &rest = std::nullopt,
                   const std::optional<std::vector<float>> &edge_ke = std::nullopt,
                   const std::optional<std::vector<float>> &edge_kd = std::nullopt);

    /// Helper to create a regular planar cloth grid
    //
    //        Creates a rectangular grid of particles with FEM triangles and bending elements
    //        automatically.
    /// \param pos The position of the cloth in world space
    /// \param rot The orientation of the cloth in world space
    /// \param vel The velocity of the cloth in world space
    /// \param dim_x The number of rectangular cells along the x-axis
    /// \param dim_y The number of rectangular cells along the y-axis
    /// \param cell_x The width of each cell in the x-direction
    /// \param cell_y The width of each cell in the y-direction
    /// \param mass The mass of each particle
    /// \param reverse_winding Flip the winding of the mesh
    /// \param fix_left Make the left-most edge of particles kinematic (fixed in place)
    /// \param fix_right Make the right-most edge of particles kinematic
    /// \param fix_top Make the top-most edge of particles kinematic
    /// \param fix_bottom Make the bottom-most edge of particles kinematic
    /// \param tri_ke
    /// \param tri_ka
    /// \param tri_kd
    /// \param tri_drag
    /// \param tri_lift
    /// \param edge_ke
    /// \param edge_kd
    /// \param add_springs
    /// \param spring_ke
    /// \param spring_kd
    void add_cloth_grid(const Vector3F &pos,
                        const QuaternionF &rot,
                        const Vector3F &vel,
                        int dim_x,
                        int dim_y,
                        float cell_x,
                        float cell_y,
                        float mass,
                        bool reverse_winding = false,
                        bool fix_left = false,
                        bool fix_right = false,
                        bool fix_top = false,
                        bool fix_bottom = false,
                        float tri_ke = default_tri_ke,
                        float tri_ka = default_tri_ka,
                        float tri_kd = default_tri_kd,
                        float tri_drag = default_tri_drag,
                        float tri_lift = default_tri_lift,
                        float edge_ke = default_edge_ke,
                        float edge_kd = default_edge_kd,
                        bool add_springs = false,
                        float spring_ke = default_spring_ke,
                        float spring_kd = default_spring_kd);

    /// Helper to create a cloth model from a regular triangle mesh
    //
    //        Creates one FEM triangle element and one bending element for every face
    //        and edge in the input triangle mesh
    /// \param pos The position of the cloth in world space
    /// \param rot The orientation of the cloth in world space
    /// \param scale
    /// \param vel The velocity of the cloth in world space
    /// \param vertices A list of vertex positions
    /// \param indices A list of triangle indices, 3 entries per-face
    /// \param density The density per-area of the mesh
    /// \param tri_ke
    /// \param tri_ka
    /// \param tri_kd
    /// \param tri_drag
    /// \param tri_lift
    /// \param edge_ke
    /// \param edge_kd
    /// \param add_springs
    /// \param spring_ke
    /// \param spring_kd
    /// \remark The mesh should be two manifold.
    void add_cloth_mesh(const Vector3F &pos,
                        const QuaternionF &rot,
                        float scale,
                        const Vector3F &vel,
                        const std::vector<Vector3F> &vertices,
                        const std::vector<int> &indices,
                        float density,
                        float tri_ke = default_tri_ke,
                        float tri_ka = default_tri_ka,
                        float tri_kd = default_tri_kd,
                        float tri_drag = default_tri_drag,
                        float tri_lift = default_tri_lift,
                        float edge_ke = default_edge_ke,
                        float edge_kd = default_edge_kd,
                        bool add_springs = false,
                        float spring_ke = default_spring_ke,
                        float spring_kd = default_spring_kd);

    void add_particle_grid(const Vector3F &pos,
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
                           float radius_mean = default_particle_radius,
                           float radius_std = 0.0);

    /// Helper to create a rectangular tetrahedral FEM grid
    //
    //        Creates a regular grid of FEM tetrahedra and surface triangles. Useful for example
    //        to create beams and sheets. Each hexahedral cell is decomposed into 5
    //        tetrahedral elements.
    /// \param pos The position of the solid in world space
    /// \param rot The orientation of the solid in world space
    /// \param vel The velocity of the solid in world space
    /// \param dim_x The number of rectangular cells along the x-axis
    /// \param dim_y The number of rectangular cells along the y-axis
    /// \param dim_z The number of rectangular cells along the z-axis
    /// \param cell_x The width of each cell in the x-direction
    /// \param cell_y The width of each cell in the y-direction
    /// \param cell_z The width of each cell in the z-direction
    /// \param density The density of each particle
    /// \param k_mu The first elastic Lame parameter
    /// \param k_lambda The second elastic Lame parameter
    /// \param k_damp The damping stiffness
    /// \param fix_left Make the left-most edge of particles kinematic (fixed in place)
    /// \param fix_right Make the right-most edge of particles kinematic
    /// \param fix_top Make the top-most edge of particles kinematic
    /// \param fix_bottom Make the bottom-most edge of particles kinematic
    /// \param tri_ke
    /// \param tri_ka
    /// \param tri_kd
    /// \param tri_drag
    /// \param tri_lift
    void add_soft_grid(const Vector3F &pos,
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
                       bool fix_left = false,
                       bool fix_right = false,
                       bool fix_top = false,
                       bool fix_bottom = false,
                       float tri_ke = default_tri_ke,
                       float tri_ka = default_tri_ka,
                       float tri_kd = default_tri_kd,
                       float tri_drag = default_tri_drag,
                       float tri_lift = default_tri_lift);

    /// Helper to create a tetrahedral model from an input tetrahedral mesh
    /// \param pos The position of the solid in world space
    /// \param rot The orientation of the solid in world space
    /// \param scale
    /// \param vel The velocity of the solid in world space
    /// \param vertices A list of vertex positions
    /// \param indices A list of tetrahedron indices, 4 entries per-element
    /// \param density The density per-area of the mesh
    /// \param k_mu The first elastic Lame parameter
    /// \param k_lambda The second elastic Lame parameter
    /// \param k_damp The damping stiffness
    /// \param tri_ke
    /// \param tri_ka
    /// \param tri_kd
    /// \param tri_drag
    /// \param tri_lift
    void add_soft_mesh(const Vector3F &pos,
                       const QuaternionF &rot,
                       float scale,
                       const Vector3F &vel,
                       const std::vector<Vector3F> &vertices,
                       const std::vector<int> &indices,
                       float density,
                       float k_mu,
                       float k_lambda,
                       float k_damp,
                       float tri_ke = default_tri_ke,
                       float tri_ka = default_tri_ka,
                       float tri_kd = default_tri_kd,
                       float tri_drag = default_tri_drag,
                       float tri_lift = default_tri_lift);
};
}// namespace vox