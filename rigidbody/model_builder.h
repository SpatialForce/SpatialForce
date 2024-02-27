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

enum class UpAxis : int {
    X,
    Y,
    Z
};

enum class PARTICLE_FLAG : int {
    ACTIVE = 1 << 0
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
    Tensor1<Vector3F> particle_q;
    Tensor1<Vector3F> particle_qd;
    Tensor1<float> particle_mass;
    Tensor1<float> particle_radius;
    Tensor1<int> particle_flags;
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

    /// Adds a sphere collision shape to a body.
    /// \param body The index of the parent body this shape belongs to (use -1 for static shapes)
    /// \param pos The location of the shape with respect to the parent frame
    /// \param rot The rotation of the shape with respect to the parent frame
    /// \param radius The radius of the sphere
    /// \param density The density of the shape
    /// \param ke The contact elastic stiffness
    /// \param kd The contact damping stiffness
    /// \param kf The contact friction stiffness
    /// \param mu The coefficient of friction
    /// \param restitution The coefficient of restitution
    /// \param is_solid Whether the sphere is solid or hollow
    /// \param thickness Thickness to use for computing inertia of a hollow sphere, and for collision handling
    /// \param has_ground_collision If True, the mesh will collide with the ground plane if `Model.ground` is True
    /// \return The index of the added shape
    size_t add_shape_sphere(int body,
                            const Vector3F &pos = {0.0, 0.0, 0.0},
                            const QuaternionF &rot = QuaternionF(0.0, 0.0, 0.0, 1.0),
                            float radius = 1.0,
                            float density = default_shape_density,
                            float ke = default_shape_ke,
                            float kd = default_shape_kd,
                            float kf = default_shape_kf,
                            float mu = default_shape_mu,
                            float restitution = default_shape_restitution,
                            bool is_solid = true,
                            float thickness = default_geo_thickness,
                            bool has_ground_collision = true);

    /// Adds a box collision shape to a body.
    /// \param body The index of the parent body this shape belongs to (use -1 for static shapes)
    /// \param pos The location of the shape with respect to the parent frame
    /// \param rot The rotation of the shape with respect to the parent frame
    /// \param hx The half-extent along the x-axis
    /// \param hy The half-extent along the y-axis
    /// \param hz  The half-extent along the z-axis
    /// \param density The density of the shape
    /// \param ke The contact elastic stiffness
    /// \param kd The contact damping stiffness
    /// \param kf The contact friction stiffness
    /// \param mu The coefficient of friction
    /// \param restitution The coefficient of restitution
    /// \param is_solid Whether the box is solid or hollow
    /// \param thickness Thickness to use for computing inertia of a hollow box, and for collision handling
    /// \param has_ground_collision If True, the mesh will collide with the ground plane if `Model.ground` is True
    /// \return The index of the added shape
    size_t add_shape_box(int body,
                         const Vector3F &pos = Vector3F(0.0, 0.0, 0.0),
                         const QuaternionF &rot = QuaternionF(0.0, 0.0, 0.0, 1.0),
                         float hx = 0.5,
                         float hy = 0.5,
                         float hz = 0.5,
                         float density = default_shape_density,
                         float ke = default_shape_ke,
                         float kd = default_shape_kd,
                         float kf = default_shape_kf,
                         float mu = default_shape_mu,
                         float restitution = default_shape_restitution,
                         bool is_solid = true,
                         float thickness = default_geo_thickness,
                         bool has_ground_collision = true);

    /// Adds a capsule collision shape to a body.
    /// \param body The index of the parent body this shape belongs to (use -1 for static shapes)
    /// \param pos The location of the shape with respect to the parent frame
    /// \param rot The rotation of the shape with respect to the parent frame
    /// \param radius The radius of the capsule
    /// \param half_height The half length of the center cylinder along the up axis
    /// \param axis The axis along which the capsule is aligned (0=x, 1=y, 2=z)
    /// \param density The density of the shape
    /// \param ke The contact elastic stiffness
    /// \param kd The contact damping stiffness
    /// \param kf The contact friction stiffness
    /// \param mu The coefficient of friction
    /// \param restitution The coefficient of restitution
    /// \param is_solid Whether the capsule is solid or hollow
    /// \param thickness Thickness to use for computing inertia of a hollow capsule, and for collision handling
    /// \param has_ground_collision If True, the mesh will collide with the ground plane if `Model.ground` is True
    /// \return The index of the added shape
    size_t add_shape_capsule(int body,
                             const Vector3F &pos = Vector3F(0.0, 0.0, 0.0),
                             const QuaternionF &rot = QuaternionF(0.0, 0.0, 0.0, 1.0),
                             float radius = 1.0,
                             float half_height = 0.5,
                             UpAxis axis = UpAxis::Y,
                             float density = default_shape_density,
                             float ke = default_shape_ke,
                             float kd = default_shape_kd,
                             float kf = default_shape_kf,
                             float mu = default_shape_mu,
                             float restitution = default_shape_restitution,
                             bool is_solid = true,
                             float thickness = default_geo_thickness,
                             bool has_ground_collision = true);

    /// Adds a cylinder collision shape to a body.
    /// \param body The index of the parent body this shape belongs to (use -1 for static shapes)
    /// \param pos The location of the shape with respect to the parent frame
    /// \param rot The rotation of the shape with respect to the parent frame
    /// \param radius The radius of the cylinder
    /// \param half_height The half length of the cylinder along the up axis
    /// \param up_axis The axis along which the cylinder is aligned (0=x, 1=y, 2=z)
    /// \param density The density of the shape
    /// \param ke The contact elastic stiffness
    /// \param kd The contact damping stiffness
    /// \param kf The contact friction stiffness
    /// \param mu The coefficient of friction
    /// \param restitution The coefficient of restitution
    /// \param is_solid Whether the cylinder is solid or hollow
    /// \param thickness Thickness to use for computing inertia of a hollow cylinder, and for collision handling
    /// \param has_ground_collision If True, the mesh will collide with the ground plane if `Model.ground` is True
    /// \return The index of the added shape
    size_t add_shape_cylinder(int body,
                              const Vector3F &pos = Vector3F(0.0, 0.0, 0.0),
                              const QuaternionF &rot = QuaternionF(0.0, 0.0, 0.0, 1.0),
                              float radius = 1.0,
                              float half_height = 0.5,
                              UpAxis up_axis = UpAxis::Y,
                              float density = default_shape_density,
                              float ke = default_shape_ke,
                              float kd = default_shape_kd,
                              float kf = default_shape_kf,
                              float mu = default_shape_mu,
                              float restitution = default_shape_restitution,
                              bool is_solid = true,
                              float thickness = default_geo_thickness,
                              bool has_ground_collision = true);

    /// Adds a cone collision shape to a body.
    /// \param body The index of the parent body this shape belongs to (use -1 for static shapes)
    /// \param pos The location of the shape with respect to the parent frame
    /// \param rot The rotation of the shape with respect to the parent frame
    /// \param radius The radius of the cone
    /// \param half_height The half length of the cone along the up axis
    /// \param up_axis The axis along which the cone is aligned (0=x, 1=y, 2=z)
    /// \param density The density of the shape
    /// \param ke The contact elastic stiffness
    /// \param kd The contact damping stiffness
    /// \param kf The contact friction stiffness
    /// \param mu The coefficient of friction
    /// \param restitution The coefficient of restitution
    /// \param is_solid Whether the cone is solid or hollow
    /// \param thickness Thickness to use for computing inertia of a hollow cone, and for collision handling
    /// \param has_ground_collision If True, the mesh will collide with the ground plane if `Model.ground` is True
    /// \return The index of the added shape
    size_t add_shape_cone(int body,
                          const Vector3F &pos = Vector3F(0.0, 0.0, 0.0),
                          const QuaternionF &rot = QuaternionF(0.0, 0.0, 0.0, 1.0),
                          float radius = 1.0,
                          float half_height = 0.5,
                          UpAxis up_axis = UpAxis::Y,
                          float density = default_shape_density,
                          float ke = default_shape_ke,
                          float kd = default_shape_kd,
                          float kf = default_shape_kf,
                          float mu = default_shape_mu,
                          float restitution = default_shape_restitution,
                          bool is_solid = true,
                          float thickness = default_geo_thickness,
                          bool has_ground_collision = true);

    /// Adds a triangle mesh collision shape to a body.
    /// \param body The index of the parent body this shape belongs to (use -1 for static shapes)
    /// \param pos The location of the shape with respect to the parent frame
    /// \param rot The rotation of the shape with respect to the parent frame
    /// \param mesh The mesh object
    /// \param scale Scale to use for the collider
    /// \param density The density of the shape
    /// \param ke The contact elastic stiffness
    /// \param kd The contact damping stiffness
    /// \param kf The contact friction stiffness
    /// \param mu The coefficient of friction
    /// \param restitution The coefficient of restitution
    /// \param is_solid If True, the mesh is solid, otherwise it is a hollow surface with the given wall thickness
    /// \param thickness Thickness to use for computing inertia of a hollow mesh, and for collision handling
    /// \param has_ground_collision If True, the mesh will collide with the ground plane if `Model.ground` is True
    /// \return The index of the added shape
    size_t add_shape_mesh(int body,
                          const Vector3F &pos = Vector3F(0.0, 0.0, 0.0),
                          const QuaternionF &rot = QuaternionF(0.0, 0.0, 0.0, 1.0),
                          std::optional<int> mesh = std::nullopt,
                          const Vector3F &scale = Vector3F(1.0, 1.0, 1.0),
                          float density = default_shape_density,
                          float ke = default_shape_ke,
                          float kd = default_shape_kd,
                          float kf = default_shape_kf,
                          float mu = default_shape_mu,
                          float restitution = default_shape_restitution,
                          bool is_solid = true,
                          float thickness = default_geo_thickness,
                          bool has_ground_collision = true);

    /// Adds SDF collision shape to a body.
    /// \param body The index of the parent body this shape belongs to (use -1 for static shapes)
    /// \param pos The location of the shape with respect to the parent frame
    /// \param rot The rotation of the shape with respect to the parent frame
    /// \param sdf The sdf object
    /// \param scale Scale to use for the collider
    /// \param density The density of the shape
    /// \param ke The contact elastic stiffness
    /// \param kd The contact damping stiffness
    /// \param kf The contact friction stiffness
    /// \param mu The coefficient of friction
    /// \param restitution The coefficient of restitution
    /// \param is_solid If True, the mesh is solid, otherwise it is a hollow surface with the given wall thickness
    /// \param thickness Thickness to use for computing inertia of a hollow mesh, and for collision handling
    /// \param has_ground_collision If True, the mesh will collide with the ground plane if `Model.ground` is True
    /// \return The index of the added shape
    size_t add_shape_sdf(int body,
                         const Vector3F &pos = Vector3F(0.0, 0.0, 0.0),
                         const QuaternionF &rot = QuaternionF(0.0, 0.0, 0.0, 1.0),
                         std::optional<int> sdf = std::nullopt,
                         const Vector3F &scale = Vector3F(1.0, 1.0, 1.0),
                         float density = default_shape_density,
                         float ke = default_shape_ke,
                         float kd = default_shape_kd,
                         float kf = default_shape_kf,
                         float mu = default_shape_mu,
                         float restitution = default_shape_restitution,
                         bool is_solid = true,
                         float thickness = default_geo_thickness,
                         bool has_ground_collision = true);

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

    /// incrementally updates rigid body mass with additional mass and inertia expressed at a local to the body
    void _update_body_mass(int i, float m, const Matrix3x3F &I, const Vector3F &p, const QuaternionF &q);

    /// Creates a ground plane for the world. If the normal is not specified,
    //        the up_vector of the ModelBuilder is used.
    void set_ground_plane(const Vector3F &normal = Vector3F(0, 1, 0),
                          float offset = 0.0,
                          float ke = default_shape_ke,
                          float kd = default_shape_kd,
                          float kf = default_shape_kf,
                          float mu = default_shape_mu,
                          float restitution = default_shape_restitution);

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