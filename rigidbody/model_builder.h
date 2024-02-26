//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "tensor/tensor.h"
#include <unordered_set>
#include <unordered_map>

namespace vox {
// A helper class for building simulation models at runtime.
//
//    Use the ModelBuilder to construct a simulation scene. The ModelBuilder
//    and builds the scene representation using standard Python data structures (lists),
//    this means it is not differentiable. Once :func:`finalize()`
//    has been called the ModelBuilder transfers all data to Warp tensors and returns
//    an object that may be used for simulation.
class ModelBuilder {
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

    // rigid shape contact material properties
    static constexpr float default_shape_ke = 1.0e5;
    static constexpr float default_shape_kd = 1000.0;
    static constexpr float default_shape_kf = 1000.0;
    static constexpr float default_shape_mu = 0.5;
    static constexpr float default_shape_restitution = 0.0;
    static constexpr float default_shape_density = 1000.0;

    // joint settings
    static constexpr float default_joint_limit_ke = 100.0;
    static constexpr float default_joint_limit_kd = 1.0;

    // geo settings
    static constexpr float default_geo_thickness = 1e-5;

    int num_envs = 0;

    // particles
    Tensor1<float> particle_q;
    Tensor1<float> particle_qd;
    Tensor1<float> particle_mass;
    Tensor1<float> particle_radius;
    Tensor1<float> particle_flags;
    float particle_max_velocity = 1e5;

    // shapes (each shape has an entry in these arrays)
    // transform from shape to body
    Tensor1<float> shape_transform;
    // maps from shape index to body index
    Tensor1<float> shape_body;
    Tensor1<float> shape_geo_type;
    Tensor1<float> shape_geo_scale;
    Tensor1<float> shape_geo_src;
    Tensor1<float> shape_geo_is_solid;
    Tensor1<float> shape_geo_thickness;
    Tensor1<float> shape_material_ke;
    Tensor1<float> shape_material_kd;
    Tensor1<float> shape_material_kf;
    Tensor1<float> shape_material_mu;
    Tensor1<float> shape_material_restitution;
    // collision groups within collisions are handled
    Tensor1<float> shape_collision_group;
    Tensor1<float> shape_collision_group_map;
    int last_collision_group = 0;
    // radius to use for broadphase collision checking
    Tensor1<float> shape_collision_radius;
    // whether the shape collides with the ground
    Tensor1<float> shape_ground_collision;

    // filtering to ignore certain collision pairs
    std::unordered_set<int> shape_collision_filter_pairs{};

    // geometry
    Tensor1<float> geo_meshes;
    Tensor1<float> geo_sdfs;

    // springs
    Tensor1<float> spring_indices;
    Tensor1<float> spring_rest_length;
    Tensor1<float> spring_stiffness;
    Tensor1<float> spring_damping;
    Tensor1<float> spring_control;

    // triangles
    Tensor1<float> tri_indices;
    Tensor1<float> tri_poses;
    Tensor1<float> tri_activations;
    Tensor1<float> tri_materials;

    // edges (bending)
    Tensor1<float> edge_indices;
    Tensor1<float> edge_rest_angle;
    Tensor1<float> edge_bending_properties;

    // tetrahedra
    Tensor1<float> tet_indices;
    Tensor1<float> tet_poses;
    Tensor1<float> tet_activations;
    Tensor1<float> tet_materials;

    // muscles
    Tensor1<float> muscle_start;
    Tensor1<float> muscle_params;
    Tensor1<float> muscle_activation;
    Tensor1<float> muscle_bodies;
    Tensor1<float> muscle_points;

    // rigid bodies
    Tensor1<float> body_mass;
    Tensor1<float> body_inertia;
    Tensor1<float> body_inv_mass;
    Tensor1<float> body_inv_inertia;
    Tensor1<float> body_com;
    Tensor1<float> body_q;
    Tensor1<float> body_qd;
    Tensor1<float> body_name;
    // mapping from body to shapes
    Tensor1<float> body_shapes;

    // rigid joints
    Tensor1<float> joint;
    // index of the parent body  (constant)
    Tensor1<float> joint_parent;
    // mapping from joint to parent bodies
    Tensor1<float> joint_parents;
    // index of the child body (constant)
    Tensor1<float> joint_child;
    // joint axis in child joint frame (constant)
    Tensor1<float> joint_axis;
    // frame of joint in parent (constant)
    Tensor1<float> joint_X_p;
    // frame of child com (in child coordinates)  (constant)
    Tensor1<float> joint_X_c;
    Tensor1<float> joint_q;
    Tensor1<float> joint_qd;

    Tensor1<float> joint_type;
    Tensor1<float> joint_name;
    Tensor1<float> joint_armature;
    Tensor1<float> joint_target;
    Tensor1<float> joint_target_ke;
    Tensor1<float> joint_target_kd;
    Tensor1<float> joint_axis_mode;
    Tensor1<float> joint_limit_lower;
    Tensor1<float> joint_limit_upper;
    Tensor1<float> joint_limit_ke;
    Tensor1<float> joint_limit_kd;
    Tensor1<float> joint_act;

    Tensor1<float> joint_twist_lower;
    Tensor1<float> joint_twist_upper;

    Tensor1<float> joint_linear_compliance;
    Tensor1<float> joint_angular_compliance;
    Tensor1<float> joint_enabled;

    Tensor1<size_t> joint_q_start;
    Tensor1<size_t> joint_qd_start;
    Tensor1<size_t> joint_axis_start;
    Tensor1<float> joint_axis_dim;
    Tensor1<size_t> articulation_start;

    int joint_dof_count = 0;
    int joint_coord_count = 0;
    int joint_axis_total_count = 0;

    Vector3F up_vector;
    Vector3F up_axis;
    float gravity;
    // indicates whether a ground plane has been created
    bool _ground_created{false};
    // constructor parameters for ground plane shape
    struct GroundParams {
        std::array<float, 3> plane{};
        float width{};
        float length{};
        float ke{default_shape_ke};
        float kd{default_shape_kd};
        float kf{default_shape_kf};
        float mu{default_shape_mu};
        float restitution{default_shape_restitution};
    };
    GroundParams _ground_params;

    // Maximum number of soft contacts that can be registered
    int soft_contact_max = 64 * 1024;

    // contacts to be generated within the given distance margin to be generated at
    // every simulation substep (can be 0 if only one PBD solver iteration is used)
    float rigid_contact_margin = 0.1;
    // torsional friction coefficient (only considered by XPBD so far)
    float rigid_contact_torsional_friction = 0.5;
    // rolling friction coefficient (only considered by XPBD so far)
    float rigid_contact_rolling_friction = 0.001;

    // number of rigid contact points to allocate in the model during self.finalize() per environment
    // if setting is None, the number of worst-case number of contacts will be calculated in self.finalize()
    int num_rigid_contacts_per_env{};

public:
    explicit ModelBuilder(const Vector3F &up_vector = {0.0, 1.0, 0.0}, float gravity = -9.80665);

    size_t shape_count();

    size_t body_count();

    size_t joint_count();

    size_t joint_axis_count();

    size_t particle_count();

    size_t tri_count();

    size_t tet_count();

    size_t edge_count();

    size_t spring_count();

    size_t muscle_count();

    size_t articulation_count();

    void add_articulation();

    void add_builder();

    void add_body();

    void add_joint();

    void add_joint_revolute();

    void add_joint_prismatic();

    void add_joint_ball();

    void add_joint_fixed();

    void add_joint_free();

    void add_joint_distance();

    void add_joint_universal();

    void add_joint_compound();

    void add_joint_d6();

    void collapse_fixed_joints();

    void add_muscle();

    void add_shape_plane();

    void add_shape_sphere();

    void add_shape_box();

    void add_shape_capsule();

    void add_shape_cylinder();

    void add_shape_cone();

    void add_shape_mesh();

    void add_shape_sdf();

    void _shape_radius();

    void _add_shape();

    void add_particle();

    void add_spring();

    void add_triangle();

    void add_triangles();

    void add_tetrahedron();

    void add_edge();

    void add_edges();

    void add_cloth_grid();

    void add_cloth_mesh();

    void add_particle_grid();

    void add_soft_grid();

    void add_soft_mesh();

    void _update_body_mass();

    void set_ground_plane();

    void _create_ground_plane();

    /// Convert this builder object to a concrete model for simulation.
    ///
    /// After building simulation elements this method should be called to transfer
    /// all data to device memory ready for simulation.
    /// \param index The simulation device to use
    /// \return A model object.
    void finalize(uint32_t index = 0);
};
}// namespace vox