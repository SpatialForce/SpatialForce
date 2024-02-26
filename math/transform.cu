//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "transform.h"

namespace vox {

// MARK: Orientation2

CUDA_CALLABLE Orientation<2>::Orientation() : Orientation(0.0) {}

CUDA_CALLABLE Orientation<2>::Orientation(float angleInRadian) {
    setRotation(angleInRadian);
}

CUDA_CALLABLE float Orientation<2>::rotation() const { return _angle; }

CUDA_CALLABLE void Orientation<2>::setRotation(float angleInRadian) {
    _angle = angleInRadian;
    _cosAngle = ::cosf(angleInRadian);
    _sinAngle = ::sinf(angleInRadian);
}

CUDA_CALLABLE Vector2F Orientation<2>::toLocal(const Vector2F &pointInWorld) const {
    // Convert to the local frame
    return Vector2F(_cosAngle * pointInWorld.x + _sinAngle * pointInWorld.y,
                    -_sinAngle * pointInWorld.x + _cosAngle * pointInWorld.y);
}

CUDA_CALLABLE Vector2F Orientation<2>::toWorld(const Vector2F &pointInLocal) const {
    // Convert to the world frame
    return Vector2F(_cosAngle * pointInLocal.x - _sinAngle * pointInLocal.y,
                    _sinAngle * pointInLocal.x + _cosAngle * pointInLocal.y);
}

// MARK: Orientation3

CUDA_CALLABLE Orientation<3>::Orientation() = default;

CUDA_CALLABLE Orientation<3>::Orientation(const QuaternionF &quat) { setRotation(quat); }

CUDA_CALLABLE const QuaternionF &Orientation<3>::rotation() const { return _quat; }

CUDA_CALLABLE void Orientation<3>::setRotation(const QuaternionF &quat) {
    _quat = quat;
    _rotationMat3 = quat.matrix3();
    _inverseRotationMat3 = quat.inverse().matrix3();
}

CUDA_CALLABLE Vector3F Orientation<3>::toLocal(const Vector3F &pointInWorld) const {
    return _inverseRotationMat3 * pointInWorld;
}

CUDA_CALLABLE Vector3F Orientation<3>::toWorld(const Vector3F &pointInLocal) const {
    return _rotationMat3 * pointInLocal;
}

// MARK: Transform2 and 3

template<size_t N>
CUDA_CALLABLE Transform<N>::Transform() = default;

template<size_t N>
CUDA_CALLABLE Transform<N>::Transform(const Vector<float, N> &translation,
                                      const Orientation<N> &orientation) {
    setTranslation(translation);
    setOrientation(orientation);
}

template<size_t N>
CUDA_CALLABLE const Vector<float, N> &Transform<N>::translation() const {
    return _translation;
}

template<size_t N>
CUDA_CALLABLE void Transform<N>::setTranslation(const Vector<float, N> &translation) {
    _translation = translation;
}

template<size_t N>
CUDA_CALLABLE const Orientation<N> &Transform<N>::orientation() const {
    return _orientation;
}

template<size_t N>
CUDA_CALLABLE void Transform<N>::setOrientation(const Orientation<N> &orientation) {
    _orientation = orientation;
}

template<size_t N>
CUDA_CALLABLE Vector<float, N> Transform<N>::toLocal(const Vector<float, N> &pointInWorld) const {
    return _orientation.toLocal(pointInWorld - _translation);
}

template<size_t N>
CUDA_CALLABLE Vector<float, N> Transform<N>::toLocalDirection(
    const Vector<float, N> &dirInWorld) const {
    return _orientation.toLocal(dirInWorld);
}

template<size_t N>
CUDA_CALLABLE Ray<float, N> Transform<N>::toLocal(const Ray<float, N> &rayInWorld) const {
    return Ray<float, N>(toLocal(rayInWorld.origin),
                         toLocalDirection(rayInWorld.direction));
}

template<size_t N>
CUDA_CALLABLE BoundingBox<float, N> Transform<N>::toLocal(
    const BoundingBox<float, N> &bboxInWorld) const {
    BoundingBox<float, N> bboxInLocal;
    int numCorners = 2 << N;
    for (int i = 0; i < numCorners; ++i) {
        auto cornerInLocal = toLocal(bboxInWorld.corner(i));
        bboxInLocal.lowerCorner = min(bboxInLocal.lowerCorner, cornerInLocal);
        bboxInLocal.upperCorner = max(bboxInLocal.upperCorner, cornerInLocal);
    }
    return bboxInLocal;
}

template<size_t N>
CUDA_CALLABLE Vector<float, N> Transform<N>::toWorld(const Vector<float, N> &pointInLocal) const {
    return _orientation.toWorld(pointInLocal) + _translation;
}

template<size_t N>
CUDA_CALLABLE Vector<float, N> Transform<N>::toWorldDirection(
    const Vector<float, N> &dirInLocal) const {
    return _orientation.toWorld(dirInLocal);
}

template<size_t N>
CUDA_CALLABLE Ray<float, N> Transform<N>::toWorld(const Ray<float, N> &rayInLocal) const {
    return Ray<float, N>(toWorld(rayInLocal.origin),
                         toWorldDirection(rayInLocal.direction));
}

template<size_t N>
CUDA_CALLABLE BoundingBox<float, N> Transform<N>::toWorld(const BoundingBox<float, N> &bboxInLocal) const {
    BoundingBox<float, N> bboxInWorld;
    int numCorners = 2 << N;
    for (int i = 0; i < numCorners; ++i) {
        auto cornerInWorld = toWorld(bboxInLocal.corner(i));
        bboxInWorld.lowerCorner = min(bboxInWorld.lowerCorner, cornerInWorld);
        bboxInWorld.upperCorner = max(bboxInWorld.upperCorner, cornerInWorld);
    }
    return bboxInWorld;
}

template class Transform<2>;
template class Transform<3>;

}// namespace vox