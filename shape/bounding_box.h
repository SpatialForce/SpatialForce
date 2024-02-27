//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "bounding_box.h"
#include "../math/matrix.h"
#include "ray.h"

#include "../../../../../usr/include/c++/11/limits"

namespace vox {

//!
//! \brief  Box-ray intersection result.
//!
//! \tparam T   The value type.
//!
template<typename T>
struct BoundingBoxRayIntersection {
    //! True if the box and ray intersects.
    bool isIntersecting = false;

    //! Distance to the first intersection point.
    T tNear = std::numeric_limits<T>::max();

    //! Distance to the second (and the last) intersection point.
    T tFar = std::numeric_limits<T>::max();
};

//!
//! \brief  N-D axis-aligned bounding box class.
//!
//! \tparam T   Real number type.
//! \tparam N   Dimension.
//!
template<typename T, size_t N>
class BoundingBox {
public:
    static_assert(N > 0, "Dimension should be greater than 0");
    static_assert(
        std::is_floating_point<T>::value,
        "BoundingBox only can be instantiated with floating point types");

    using VectorType = Vector<T, N>;
    using RayType = Ray<T, N>;

    //! Lower corner of the bounding box.
    VectorType lowerCorner;

    //! Upper corner of the bounding box.
    VectorType upperCorner;

    //! Default constructor.
    CUDA_CALLABLE BoundingBox();

    //! Constructs a box that tightly covers two points.
    CUDA_CALLABLE BoundingBox(const VectorType &point1, const VectorType &point2);

    //! Constructs a box with other box instance.
    CUDA_CALLABLE BoundingBox(const BoundingBox &other);

    //! Returns the size of the box.
    CUDA_CALLABLE VectorType size() const;

    //! Returns width of the box.
    CUDA_CALLABLE T width() const;

    //! Returns height of the box.
    template<typename U = T>
    CUDA_CALLABLE std::enable_if_t<(N > 1), U> height() const;

    //! Returns depth of the box.
    template<typename U = T>
    CUDA_CALLABLE std::enable_if_t<(N > 2), U> depth() const;

    //! Returns length of the box in given axis.
    CUDA_CALLABLE T length(size_t axis);

    //! Returns true of this box and other box overlaps.
    CUDA_CALLABLE bool overlaps(const BoundingBox &other) const;

    CUDA_CALLABLE//! Returns true if the input vector is inside of this box.
        bool
        contains(const VectorType &point) const;

    //! Returns true if the input ray is intersecting with this box.
    CUDA_CALLABLE bool intersects(const RayType &ray) const;

    //! Returns intersection.isIntersecting = true if the input ray is
    //! intersecting with this box. If interesects, intersection.tNear is
    //! assigned with distant to the closest intersecting point, and
    //! intersection.tFar with furthest.
    CUDA_CALLABLE BoundingBoxRayIntersection<T> closestIntersection(const RayType &ray) const;

    //! Returns the mid-point of this box.
    CUDA_CALLABLE VectorType midPoint() const;

    //! Returns diagonal length of this box.
    CUDA_CALLABLE T diagonalLength() const;

    //! Returns squared diagonal length of this box.
    CUDA_CALLABLE T diagonalLengthSquared() const;

    //! Resets this box to initial state (min=infinite, max=-infinite).
    CUDA_CALLABLE void reset();

    //! Merges this and other point.
    CUDA_CALLABLE void merge(const VectorType &point);

    //! Merges this and other box.
    CUDA_CALLABLE void merge(const BoundingBox &other);

    //! Expands this box by given delta to all direction.
    //! If the width of the box was x, expand(y) will result a box with
    //! x+y+y width.
    CUDA_CALLABLE void expand(T delta);

    //! Returns corner position. Index starts from x-first order.
    CUDA_CALLABLE VectorType corner(size_t idx) const;

    //! Returns the clamped point.
    CUDA_CALLABLE VectorType clamp(const VectorType &pt) const;

    //! Returns true if the box is empty.
    CUDA_CALLABLE bool isEmpty() const;

    //! Returns box with different value type.
    template<typename U>
    CUDA_CALLABLE BoundingBox<U, N> castTo() const;
};

template<typename T>
using BoundingBox2 = BoundingBox<T, 2>;

template<typename T>
using BoundingBox3 = BoundingBox<T, 3>;

using BoundingBox2F = BoundingBox2<float>;

using BoundingBox2D = BoundingBox2<double>;

using BoundingBox3F = BoundingBox3<float>;

using BoundingBox3D = BoundingBox3<double>;

using BoundingBoxRayIntersectionF = BoundingBoxRayIntersection<float>;

using BoundingBoxRayIntersectionD = BoundingBoxRayIntersection<double>;

}// namespace vox

#include "bounding_box-inl.h"