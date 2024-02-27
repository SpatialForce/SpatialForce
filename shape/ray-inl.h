//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {

template<typename T, size_t N>
CUDA_CALLABLE Ray<T, N>::Ray() {
    origin = VectorType{};
    direction = VectorType{};
    direction[0] = 1;
}

template<typename T, size_t N>
CUDA_CALLABLE Ray<T, N>::Ray(
    const VectorType &newOrigin,
    const VectorType &newDirection) : origin(newOrigin),
                                      direction(newDirection.normalized()) {
}

template<typename T, size_t N>
CUDA_CALLABLE Ray<T, N>::Ray(const Ray &other) : origin(other.origin),
                                                 direction(other.direction) {
}

template<typename T, size_t N>
CUDA_CALLABLE typename Ray<T, N>::VectorType Ray<T, N>::pointAt(T t) const {
    return origin + t * direction;
}

}// namespace vox
