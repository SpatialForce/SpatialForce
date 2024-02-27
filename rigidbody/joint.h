//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/matrix.h"
#include "math/transform.h"
#include "tensor/tensor.h"
#include <optional>

namespace vox {
// Types of joints linking rigid bodies
enum class JointType {
    JOINT_PRISMATIC,
    JOINT_REVOLUTE,
    JOINT_BALL,
    JOINT_FIXED,
    JOINT_FREE,
    JOINT_COMPOUND,
    JOINT_UNIVERSAL,
    JOINT_DISTANCE,
    JOINT_D6,
};

// Joint axis mode types
enum class JointMode {
    LIMIT,
    TARGET_POSITION,
    TARGET_VELOCITY
};

/// Describes a joint axis that can have limits and be driven towards a target.
struct JointAxis {
    /// The 3D axis that this JointAxis object describes
    Vector3F axis;
    /// The lower limit of the joint axis
    float limit_lower;
    //// The upper limit of the joint axis
    float limit_upper;
    /// The elastic stiffness of the joint axis limits
    float limit_ke = 100.0;
    /// The damping stiffness of the joint axis limits
    float limit_kd = 10.0;
    /// The target position or velocity (depending on the mode, see `Joint modes`_) of the joint axis
    float target{};
    /// The proportional gain of the joint axis target drive PD controller
    float target_ke = 0.0;
    /// The derivative gain of the joint axis target drive PD controller
    float target_kd = 0.0;
    /// The mode of the joint axis
    JointMode mode = JointMode::TARGET_POSITION;

    explicit JointAxis(const Vector3F &axis);
};

class ModelBuilder;

class JointModelBuilder {
public:
    ModelBuilder &builder;

    // joint settings
    static constexpr float default_joint_limit_ke = 100.0;
    static constexpr float default_joint_limit_kd = 1.0;

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
    Tensor1<TransformF> joint_X_p;
    // frame of child com (in child coordinates)  (constant)
    Tensor1<TransformF> joint_X_c;
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

public:
    explicit JointModelBuilder(ModelBuilder &builder);

    size_t joint_count();

    size_t joint_axis_count();

    size_t articulation_count();

    void add_articulation();

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
                     const TransformF &parent_xform = TransformF(),
                     const TransformF &child_xform = TransformF(),
                     float linear_compliance = 0.0,
                     float angular_compliance = 0.0,
                     bool collision_filter_parent = true,
                     bool enabled = true);

    /// Adds a revolute (hinge) joint to the model. It has one degree of freedom.
    /// \param parent The index of the parent body
    /// \param child The index of the child body
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param axis The axis of rotation in the parent body's local frame, can be a JointAxis object whose settings will be used instead of the other arguments
    /// \param target The target angle (in radians) of the joint
    /// \param target_ke The stiffness of the joint target
    /// \param target_kd The damping of the joint target
    /// \param mode
    /// \param limit_lower The lower limit of the joint
    /// \param limit_upper The upper limit of the joint
    /// \param limit_ke The stiffness of the joint limit
    /// \param limit_kd The damping of the joint limit
    /// \param linear_compliance The linear compliance of the joint
    /// \param angular_compliance The angular compliance of the joint
    /// \param name The name of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    size_t add_joint_revolute(int parent,
                              int child,
                              const TransformF &parent_xform,
                              const TransformF &child_xform,
                              const Vector3F &axis,
                              float target = 0.0,
                              float target_ke = 0.0,
                              float target_kd = 0.0,
                              JointMode mode = JointMode::TARGET_POSITION,
                              float limit_lower = -2 * kPiF,
                              float limit_upper = 2 * kPiF,
                              float limit_ke = default_joint_limit_ke,
                              float limit_kd = default_joint_limit_kd,
                              float linear_compliance = 0.0,
                              float angular_compliance = 0.0,
                              std::optional<std::string_view> name = std::nullopt,
                              bool collision_filter_parent = true,
                              bool enabled = true);

    /// Adds a prismatic (sliding) joint to the model. It has one degree of freedom.
    /// \param parent The index of the parent body
    /// \param child The index of the child body
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param axis The axis of rotation in the parent body's local frame, can be a JointAxis object whose settings will be used instead of the other arguments
    /// \param target The target position of the joint
    /// \param target_ke The stiffness of the joint target
    /// \param target_kd The damping of the joint target
    /// \param mode
    /// \param limit_lower The lower limit of the joint
    /// \param limit_upper The upper limit of the joint
    /// \param limit_ke The stiffness of the joint limit
    /// \param limit_kd The damping of the joint limit
    /// \param linear_compliance The linear compliance of the joint
    /// \param angular_compliance The angular compliance of the joint
    /// \param name The name of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    size_t add_joint_prismatic(int parent,
                               int child,
                               const TransformF &parent_xform,
                               const TransformF &child_xform,
                               const Vector3F &axis,
                               float target = 0.0,
                               float target_ke = 0.0,
                               float target_kd = 0.0,
                               JointMode mode = JointMode::TARGET_POSITION,
                               float limit_lower = -1e4,
                               float limit_upper = 1e4,
                               float limit_ke = default_joint_limit_ke,
                               float limit_kd = default_joint_limit_kd,
                               float linear_compliance = 0.0,
                               float angular_compliance = 0.0,
                               std::optional<std::string_view> name = std::nullopt,
                               bool collision_filter_parent = true,
                               bool enabled = true);

    /// Adds a ball (spherical) joint to the model. Its position is defined by a 4D quaternion (xyzw) and its velocity is a 3D vector.
    /// \param parent The index of the parent body
    /// \param child The index of the child body
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param linear_compliance The linear compliance of the joint
    /// \param angular_compliance The angular compliance of the joint
    /// \param name The name of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    size_t add_joint_ball(int parent,
                          int child,
                          const TransformF &parent_xform = TransformF(),
                          const TransformF &child_xform = TransformF(),
                          float linear_compliance = 0.0,
                          float angular_compliance = 0.0,
                          std::optional<std::string_view> name = std::nullopt,
                          bool collision_filter_parent = true,
                          bool enabled = true);

    /// Adds a fixed (static) joint to the model. It has no degrees of freedom.
    ///     See :meth:`collapse_fixed_joints` for a helper function that removes these fixed joints
    ///     and merges the connecting bodies to simplify the model and improve stability.
    /// \param parent The index of the parent body
    /// \param child The index of the child body
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param linear_compliance The linear compliance of the joint
    /// \param angular_compliance The angular compliance of the joint
    /// \param name The name of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    size_t add_joint_fixed(int parent,
                           int child,
                           const TransformF &parent_xform = TransformF(),
                           const TransformF &child_xform = TransformF(),
                           float linear_compliance = 0.0,
                           float angular_compliance = 0.0,
                           std::optional<std::string_view> name = std::nullopt,
                           bool collision_filter_parent = true,
                           bool enabled = true);

    /// Adds a free joint to the model.
    ///     It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation)
    ///     and 6 velocity degrees of freedom (first 3 angular and then 3 linear velocity dimensions).
    /// \param child The index of the child body
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param parent The index of the parent body (-1 by default to use the world frame, e.g. to make the child body and its children a floating-base mechanism)
    /// \param name The name of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    size_t add_joint_free(int child,
                          const TransformF &parent_xform = TransformF(),
                          const TransformF &child_xform = TransformF(),
                          int parent = -1,
                          std::optional<std::string_view> name = std::nullopt,
                          bool collision_filter_parent = true,
                          bool enabled = true);

    /// Adds a distance joint to the model. The distance joint constraints the distance between the joint anchor points on the two bodies (see :ref:`FK-IK`) it connects to the interval [`min_distance`, `max_distance`].
    //        It has 7 positional degrees of freedom (first 3 linear and then 4 angular dimensions for the orientation quaternion in `xyzw` notation)
    //        and 6 velocity degrees of freedom (first 3 angular and then 3 linear velocity dimensions).
    /// \param parent The index of the parent body
    /// \param child The index of the child body
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param min_distance The minimum distance between the bodies (no limit if negative)
    /// \param max_distance The maximum distance between the bodies (no limit if negative)
    /// \param compliance The compliance of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    /// \remark Distance joints are currently only supported in the :class:`XPBDIntegrator` at the moment.
    size_t add_joint_distance(int parent,
                              int child,
                              const TransformF &parent_xform = TransformF(),
                              const TransformF &child_xform = TransformF(),
                              float min_distance = -1.0,
                              float max_distance = 1.0,
                              float compliance = 0.0,
                              bool collision_filter_parent = true,
                              bool enabled = true);

    /// Adds a universal joint to the model. U-joints have two degrees of freedom, one for each axis.
    /// \param parent The index of the parent body
    /// \param child The index of the child body
    /// \param axis_0 The first axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
    /// \param axis_1 The second axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param linear_compliance The linear compliance of the joint
    /// \param angular_compliance The angular compliance of the joint
    /// \param name The name of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    size_t add_joint_universal(int parent,
                               int child,
                               const JointAxis &axis_0,
                               const JointAxis &axis_1,
                               const TransformF &parent_xform = TransformF(),
                               const TransformF &child_xform = TransformF(),
                               float linear_compliance = 0.0,
                               float angular_compliance = 0.0,
                               std::optional<std::string_view> name = std::nullopt,
                               bool collision_filter_parent = true,
                               bool enabled = true);

    /// Adds a compound joint to the model, which has 3 degrees of freedom, one for each axis.
    //        Similar to the ball joint (see :meth:`add_ball_joint`), the compound joint allows bodies to move in a 3D rotation relative to each other,
    //        except that the rotation is defined by 3 axes instead of a quaternion.
    //        Depending on the choice of axes, the orientation can be specified through Euler angles, e.g. `z-x-z` or `x-y-x`,
    //        or through a Tait-Bryan angle sequence, e.g. `z-y-x` or `x-y-z`.
    /// \param parent The index of the parent body
    /// \param child The index of the child body
    /// \param axis_0 The first axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
    /// \param axis_1 The second axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
    /// \param axis_2 The third axis of the joint, can be a JointAxis object whose settings will be used instead of the other arguments
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param name The name of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    size_t add_joint_compound(int parent,
                              int child,
                              const JointAxis &axis_0,
                              const JointAxis &axis_1,
                              const JointAxis &axis_2,
                              const TransformF &parent_xform = TransformF(),
                              const TransformF &child_xform = TransformF(),
                              std::optional<std::string_view> name = std::nullopt,
                              bool collision_filter_parent = true,
                              bool enabled = true);

    /// Adds a generic joint with custom linear and angular axes. The number of axes determines the number of degrees of freedom of the joint.
    /// \param parent The index of the parent body
    /// \param child The index of the child body
    /// \param linear_axes A list of linear axes
    /// \param angular_axes A list of angular axes
    /// \param name The name of the joint
    /// \param parent_xform The transform of the joint in the parent body's local frame
    /// \param child_xform The transform of the joint in the child body's local frame
    /// \param linear_compliance The linear compliance of the joint
    /// \param angular_compliance The angular compliance of the joint
    /// \param collision_filter_parent Whether to filter collisions between shapes of the parent and child bodies
    /// \param enabled Whether the joint is enabled
    /// \return The index of the added joint
    size_t add_joint_d6(int parent,
                        int child,
                        std::initializer_list<JointAxis> linear_axes = {},
                        std::initializer_list<JointAxis> angular_axes = {},
                        std::optional<std::string_view> name = std::nullopt,
                        const TransformF &parent_xform = TransformF(),
                        const TransformF &child_xform = TransformF(),
                        float linear_compliance = 0.0,
                        float angular_compliance = 0.0,
                        bool collision_filter_parent = true,
                        bool enabled = true);

    /// Removes fixed joints from the model and merges the bodies they connect.
    /// This is useful for simplifying the model for faster and more stable simulation.
    void collapse_fixed_joints();

private:
    void _add_axis_dim(const JointAxis &dim);
};

}// namespace vox