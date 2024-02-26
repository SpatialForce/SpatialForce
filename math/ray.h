//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "matrix.h"

namespace vox {

//!
//! \brief Class for N-D ray.
//!
//! \tparam T   The value type.
//! \tparam N   Dimension.
//!
template<typename T, size_t N>
class Ray final {
public:
    static_assert(N > 0, "Dimension should be greater than 0");
    static_assert(std::is_floating_point<T>::value,
                  "Ray only can be instantiated with floating point types");

    using VectorType = Vector<T, N>;

    //! The origin of the ray.
    VectorType origin;

    //! The direction of the ray.
    VectorType direction;

    //! Constructs an empty ray that points (1, 0, ...) from (0, 0, ...).
    CUDA_CALLABLE Ray();

    //! Constructs a ray with given origin and riection.
    CUDA_CALLABLE Ray(const VectorType &newOrigin, const VectorType &newDirection);

    //! Copy constructor.
    CUDA_CALLABLE Ray(const Ray &other);

    //! Returns a point on the ray at distance \p t.
    CUDA_CALLABLE VectorType pointAt(T t) const;
};

template<typename T>
using Ray2 = Ray<T, 2>;

template<typename T>
using Ray3 = Ray<T, 3>;

using Ray2F = Ray2<float>;

using Ray2D = Ray2<double>;

using Ray3F = Ray3<float>;

using Ray3D = Ray3<double>;

}// namespace vox

#include "ray-inl.h"
