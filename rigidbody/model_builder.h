//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "tensor/tensor.h"
#include "math/transform.h"
#include "math/spatial_matrix.h"
#include "joint.h"
#include <set>
#include <unordered_map>
#include <optional>

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
    std::set<std::pair<size_t, size_t>> shape_collision_filter_pairs{};

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
    Tensor1<Matrix3x3F> body_inertia;
    Tensor1<float> body_inv_mass;
    Tensor1<Matrix3x3F> body_inv_inertia;
    Tensor1<Vector3F> body_com;
    Tensor1<Transform3> body_q;
    Tensor1<SpatialVectorF> body_qd;
    Tensor1<std::string_view> body_name;
    // mapping from body to shapes
    std::unordered_map<size_t, Tensor1<size_t>> body_shapes;

    // rigid joints
    Tensor1<float> joint;
    // index of the parent body  (constant)
    Tensor1<int> joint_parent;
    // mapping from joint to parent bodies
    std::unordered_map<int, Tensor1<int>> joint_parents;
    // index of the child body (constant)
    Tensor1<int> joint_child;
    // joint axis in child joint frame (constant)
    Tensor1<Vector3F> joint_axis;
    // frame of joint in parent (constant)
    Tensor1<Transform3> joint_X_p;
    // frame of child com (in child coordinates)  (constant)
    Tensor1<Transform3> joint_X_c;
    Tensor1<float> joint_q;
    Tensor1<float> joint_qd;

    Tensor1<float> joint_type;
    Tensor1<std::string_view> joint_name;
    Tensor1<float> joint_armature;
    Tensor1<float> joint_target;
    Tensor1<float> joint_target_ke;
    Tensor1<float> joint_target_kd;
    Tensor1<JointMode> joint_axis_mode;
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
    Tensor1<std::pair<int32_t, int32_t>> joint_axis_dim;
    Tensor1<size_t> articulation_start;

    size_t joint_dof_count = 0;
    size_t joint_coord_count = 0;
    size_t joint_axis_total_count = 0;

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

    /// Adds a rigid body to the model.
    /// \param origin the location of the body in the world frame
    /// \param armature Artificial inertia added to the body
    /// \param com The center of mass of the body w.r.t its origin
    /// \param I_m The 3x3 inertia tensor of the body (specified relative to the center of mass)
    /// \param m Mass of the body
    /// \param name Name of the body (optional)
    ///
    /// \return The index of the body in the model
    ///
    /// \remark If the mass (m) is zero then the body is treated as kinematic with no dynamics
    size_t add_body(const Transform3 &origin = Transform3(),
                    float armature = 0.0,
                    const Vector3F &com = Vector3F(),
                    const Matrix3x3F &I_m = Matrix3x3F(),
                    float m = 0.0,
                    std::optional<std::string_view> name = std::nullopt);

    /// Generic method to add any type of joint to this ModelBuilder.
    /// \param type The type of joint to add (see `Joint types`_)
    /// \param parent The index of the parent body (-1 is the world)
    /// \param child The index of the child body
    /// \param linear_axes The linear axes (see :class:`JointAxis`) of the joint
    /// \param angular_axes The angular axes (see :class:`JointAxis`) of the joint
    /// \param name The name of the joint
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param linear_compliance The linear compliance of the joint
    /// \param angular_compliance The angular compliance of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    size_t add_joint(JointType type,
                   int parent,
                   int child,
                   std::initializer_list<JointAxis> linear_axes = {},
                   std::initializer_list<JointAxis> angular_axes = {},
                   std::optional<std::string_view> name = std::nullopt,
                   const Transform3& parent_xform = Transform3(),
                   const Transform3& child_xform = Transform3(),
                   float linear_compliance = 0.0,
                   float angular_compliance = 0.0,
                   bool collision_filter_parent = true,
                   bool enabled = true);

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

private:
    void _add_axis_dim(const JointAxis &dim);
};
}// namespace vox