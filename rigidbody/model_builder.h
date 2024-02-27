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
#include "muscle.h"
#include <set>
#include <unordered_map>
#include <optional>

namespace vox {
enum class GeometryType {
    GEO_SPHERE,
    GEO_BOX,
    GEO_CAPSULE,
    GEO_CYLINDER,
    GEO_CONE,
    GEO_MESH,
    GEO_SDF,
    GEO_PLANE,
    GEO_NONE
};

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
    Tensor1<TransformF> shape_transform;
    // maps from shape index to body index
    Tensor1<int> shape_body;
    Tensor1<GeometryType> shape_geo_type;
    Tensor1<float> shape_geo_scale;
    Tensor1<std::optional<int>> shape_geo_src;
    Tensor1<float> shape_geo_is_solid;
    Tensor1<float> shape_geo_thickness;
    Tensor1<float> shape_material_ke;
    Tensor1<float> shape_material_kd;
    Tensor1<float> shape_material_kf;
    Tensor1<float> shape_material_mu;
    Tensor1<float> shape_material_restitution;
    // collision groups within collisions are handled
    Tensor1<int> shape_collision_group;
    std::unordered_map<int, Tensor1<size_t>> shape_collision_group_map;
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
    MuscleModelBuilder muscle_builder;

    // rigid bodies
    Tensor1<float> body_mass;
    Tensor1<Matrix3x3F> body_inertia;
    Tensor1<float> body_inv_mass;
    Tensor1<Matrix3x3F> body_inv_inertia;
    Tensor1<Vector3F> body_com;
    Tensor1<TransformF> body_q;
    Tensor1<SpatialVectorF> body_qd;
    Tensor1<std::string_view> body_name;
    // mapping from body to shapes
    std::unordered_map<size_t, Tensor1<size_t>> body_shapes;

    // rigid joints
    JointModelBuilder joint_builder;

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

    size_t particle_count();

    size_t tri_count();

    size_t tet_count();

    size_t edge_count();

    size_t spring_count();

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
    size_t add_body(const TransformF &origin = TransformF(),
                    float armature = 0.0,
                    const Vector3F &com = Vector3F(),
                    const Matrix3x3F &I_m = Matrix3x3F(),
                    float m = 0.0,
                    std::optional<std::string_view> name = std::nullopt);

    /// Adds a plane collision shape.
    //        If pos and rot are defined, the plane is assumed to have its normal as (0, 1, 0).
    //        Otherwise, the plane equation defined through the `plane` argument is used.
    /// \param plane The plane equation in form a*x + b*y + c*z + d = 0
    /// \param pos The position of the plane in world coordinates
    /// \param rot The rotation of the plane in world coordinates
    /// \param width The extent along x of the plane (infinite if 0)
    /// \param length The extent along z of the plane (infinite if 0)
    /// \param body The body index to attach the shape to (-1 by default to keep the plane static)
    /// \param ke The contact elastic stiffness
    /// \param kd The contact damping stiffness
    /// \param kf The contact friction stiffness
    /// \param mu The coefficient of friction
    /// \param restitution The coefficient of restitution
    /// \param thickness The thickness of the plane (0 by default) for collision handling
    /// \param has_ground_collision If True, the mesh will collide with the ground plane if `Model.ground` is True
    /// \return The index of the added shape
    size_t add_shape_plane(const Vector4F &plane = Vector4F(0.0, 1.0, 0.0, 0.0),
                           const std::optional<Vector3F> &pos = std::nullopt,
                           const std::optional<QuaternionF> &rot = std::nullopt,
                           float width = 10.0,
                           float length = 10.0,
                           int body = -1,
                           float ke = default_shape_ke,
                           float kd = default_shape_kd,
                           float kf = default_shape_kf,
                           float mu = default_shape_mu,
                           float restitution = default_shape_restitution,
                           float thickness = 0.0,
                           bool has_ground_collision = false);

    void add_shape_sphere();

    void add_shape_box();

    void add_shape_capsule();

    void add_shape_cylinder();

    void add_shape_cone();

    void add_shape_mesh();

    void add_shape_sdf();

    /// Calculates the radius of a sphere that encloses the shape, used for broadphase collision detection.
    static float _shape_radius(GeometryType type, const Vector3F &scale, std::optional<int> src);

    size_t _add_shape(int body,
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
                      float thickness = default_geo_thickness,
                      bool is_solid = true,
                      int collision_group = -1,
                      bool collision_filter_parent = true,
                      bool has_ground_collision = true);

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

    void _update_body_mass(int i, float m, const Matrix3x3F& I, const Vector3F& p, const QuaternionF& q);

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