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
    CUDA_CALLABLE Orientation(float angleInRadian);

    CUDA_CALLABLE float rotation() const;
    CUDA_CALLABLE void setRotation(float angleInRadian);

    //! Rotates a point in world coordinate to the local frame.
    CUDA_CALLABLE Vector2F toLocal(const Vector2F &pointInWorld) const;

    //! Rotates a point in local space to the world coordinate.
    CUDA_CALLABLE Vector2F toWorld(const Vector2F &pointInLocal) const;

private:
    float _angle = 0.0;
    float _cosAngle = 1.0;
    float _sinAngle = 0.0;
};

template<>
class Orientation<3> {
public:
    CUDA_CALLABLE Orientation();
    CUDA_CALLABLE Orientation(const QuaternionF &quat);

    CUDA_CALLABLE const QuaternionF &rotation() const;
    CUDA_CALLABLE void setRotation(const QuaternionF &quat);

    //! Rotates a point in world coordinate to the local frame.
    CUDA_CALLABLE Vector3F toLocal(const Vector3F &pointInWorld) const;

    //! Rotates a point in local space to the world coordinate.
    CUDA_CALLABLE Vector3F toWorld(const Vector3F &pointInLocal) const;

private:
    QuaternionF _quat;
    Matrix3x3F _rotationMat3 = Matrix3x3F::makeIdentity();
    Matrix3x3F _inverseRotationMat3 = Matrix3x3F::makeIdentity();
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
    CUDA_CALLABLE Transform(const Vector<float, N> &translation,
                            const Orientation<N> &orientation);

    //! Returns the translation.
    CUDA_CALLABLE const Vector<float, N> &translation() const;

    //! Sets the traslation.
    CUDA_CALLABLE void setTranslation(const Vector<float, N> &translation);

    //! Returns the orientation.
    CUDA_CALLABLE const Orientation<N> &orientation() const;

    //! Sets the orientation.
    CUDA_CALLABLE void setOrientation(const Orientation<N> &orientation);

    //! Transforms a point in world coordinate to the local frame.
    CUDA_CALLABLE Vector<float, N> toLocal(const Vector<float, N> &pointInWorld) const;

    //! Transforms a direction in world coordinate to the local frame.
    CUDA_CALLABLE Vector<float, N> toLocalDirection(const Vector<float, N> &dirInWorld) const;

    //! Transforms a ray in world coordinate to the local frame.
    CUDA_CALLABLE Ray<float, N> toLocal(const Ray<float, N> &rayInWorld) const;

    //! Transforms a bounding box in world coordinate to the local frame.
    CUDA_CALLABLE BoundingBox<float, N> toLocal(const BoundingBox<float, N> &bboxInWorld) const;

    //! Transforms a point in local space to the world coordinate.
    CUDA_CALLABLE Vector<float, N> toWorld(const Vector<float, N> &pointInLocal) const;

    //! Transforms a direction in local space to the world coordinate.
    CUDA_CALLABLE Vector<float, N> toWorldDirection(const Vector<float, N> &dirInLocal) const;

    //! Transforms a ray in local space to the world coordinate.
    CUDA_CALLABLE Ray<float, N> toWorld(const Ray<float, N> &rayInLocal) const;

    //! Transforms a bounding box in local space to the world coordinate.
    CUDA_CALLABLE BoundingBox<float, N> toWorld(const BoundingBox<float, N> &bboxInLocal) const;

private:
    Vector<float, N> _translation;
    Orientation<N> _orientation;
};

using Transform2 = Transform<2>;
using Transform3 = Transform<3>;

}// namespace vox
