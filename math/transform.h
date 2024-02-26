//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "bounding_box.h"
#include "matrix.h"
#include "quaternion.h"
#include "ray.h"

namespace vox {

template<size_t N>
class Orientation {};

template<>
class Orientation<2> {
public:
    CUDA_CALLABLE Orientation();
    CUDA_CALLABLE Orientation(double angleInRadian);

    CUDA_CALLABLE double rotation() const;
    CUDA_CALLABLE void setRotation(double angleInRadian);

    //! Rotates a point in world coordinate to the local frame.
    CUDA_CALLABLE Vector2D toLocal(const Vector2D &pointInWorld) const;

    //! Rotates a point in local space to the world coordinate.
    CUDA_CALLABLE Vector2D toWorld(const Vector2D &pointInLocal) const;

private:
    double _angle = 0.0;
    double _cosAngle = 1.0;
    double _sinAngle = 0.0;
};

template<>
class Orientation<3> {
public:
    CUDA_CALLABLE Orientation();
    CUDA_CALLABLE Orientation(const QuaternionD &quat);

    CUDA_CALLABLE const QuaternionD &rotation() const;
    CUDA_CALLABLE void setRotation(const QuaternionD &quat);

    //! Rotates a point in world coordinate to the local frame.
    CUDA_CALLABLE Vector3D toLocal(const Vector3D &pointInWorld) const;

    //! Rotates a point in local space to the world coordinate.
    CUDA_CALLABLE Vector3D toWorld(const Vector3D &pointInLocal) const;

private:
    QuaternionD _quat;
    Matrix3x3D _rotationMat3 = Matrix3x3D::makeIdentity();
    Matrix3x3D _inverseRotationMat3 = Matrix3x3D::makeIdentity();
};

using Orientation2 = Orientation<2>;
using Orientation3 = Orientation<3>;

//!
//! \brief Represents N-D rigid body transform.
//!
template<size_t N>
class Transform {
public:
    //! Constructs identity transform.
    CUDA_CALLABLE Transform();

    //! Constructs a transform with translation and orientation.
    CUDA_CALLABLE Transform(const Vector<double, N> &translation,
                            const Orientation<N> &orientation);

    //! Returns the translation.
    CUDA_CALLABLE const Vector<double, N> &translation() const;

    //! Sets the traslation.
    CUDA_CALLABLE void setTranslation(const Vector<double, N> &translation);

    //! Returns the orientation.
    CUDA_CALLABLE const Orientation<N> &orientation() const;

    //! Sets the orientation.
    CUDA_CALLABLE void setOrientation(const Orientation<N> &orientation);

    //! Transforms a point in world coordinate to the local frame.
    CUDA_CALLABLE Vector<double, N> toLocal(const Vector<double, N> &pointInWorld) const;

    //! Transforms a direction in world coordinate to the local frame.
    CUDA_CALLABLE Vector<double, N> toLocalDirection(const Vector<double, N> &dirInWorld) const;

    //! Transforms a ray in world coordinate to the local frame.
    CUDA_CALLABLE Ray<double, N> toLocal(const Ray<double, N> &rayInWorld) const;

    //! Transforms a bounding box in world coordinate to the local frame.
    CUDA_CALLABLE BoundingBox<double, N> toLocal(const BoundingBox<double, N> &bboxInWorld) const;

    //! Transforms a point in local space to the world coordinate.
    CUDA_CALLABLE Vector<double, N> toWorld(const Vector<double, N> &pointInLocal) const;

    //! Transforms a direction in local space to the world coordinate.
    CUDA_CALLABLE Vector<double, N> toWorldDirection(const Vector<double, N> &dirInLocal) const;

    //! Transforms a ray in local space to the world coordinate.
    CUDA_CALLABLE Ray<double, N> toWorld(const Ray<double, N> &rayInLocal) const;

    //! Transforms a bounding box in local space to the world coordinate.
    CUDA_CALLABLE BoundingBox<double, N> toWorld(const BoundingBox<double, N> &bboxInLocal) const;

private:
    Vector<double, N> _translation;
    Orientation<N> _orientation;
};

using Transform2 = Transform<2>;
using Transform3 = Transform<3>;

}// namespace vox
