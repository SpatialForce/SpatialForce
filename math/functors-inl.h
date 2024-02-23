//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math_utils.h"

namespace vox {

template<typename T>
CUDA_CALLABLE constexpr T NoOp<T>::operator()(const T &a) const {
    return a;
}

template<typename T, typename U>
CUDA_CALLABLE constexpr U TypeCast<T, U>::operator()(const T &a) const {
    return static_cast<U>(a);
}

template<typename T>
CUDA_CALLABLE constexpr T Ceil<T>::operator()(const T &a) const {
    return ::ceil(a);
}

template<typename T>
CUDA_CALLABLE constexpr T Floor<T>::operator()(const T &a) const {
    return ::floor(a);
}

template<typename T>
CUDA_CALLABLE constexpr T Square<T>::operator()(const T &a) const {
    return a * a;
}

template<typename T>
CUDA_CALLABLE constexpr T RMinus<T>::operator()(const T &a, const T &b) const {
    return b - a;
}

template<typename T>
CUDA_CALLABLE constexpr T RDivides<T>::operator()(const T &a, const T &b) const {
    return b / a;
}

template<typename T>
CUDA_CALLABLE void IAdd<T>::operator()(T &a, const T &b) const {
    a += b;
}

template<typename T>
CUDA_CALLABLE void ISub<T>::operator()(T &a, const T &b) const {
    a -= b;
}

template<typename T>
CUDA_CALLABLE void IMul<T>::operator()(T &a, const T &b) const {
    a *= b;
}

template<typename T>
CUDA_CALLABLE void IDiv<T>::operator()(T &a, const T &b) const {
    a /= b;
}

template<typename T>
CUDA_CALLABLE constexpr T Min<T>::operator()(const T &a, const T &b) const {
    return ::min(a, b);
}

template<typename T>
CUDA_CALLABLE constexpr T Max<T>::operator()(const T &a, const T &b) const {
    return ::max(a, b);
}

template<typename T>
CUDA_CALLABLE constexpr T AbsMin<T>::operator()(const T &a, const T &b) const {
    return absmin(a, b);
}

template<typename T>
CUDA_CALLABLE constexpr T AbsMax<T>::operator()(const T &a, const T &b) const {
    return absmax(a, b);
}

template<typename T>
CUDA_CALLABLE constexpr bool SimilarTo<T>::operator()(const T &a, const T &b) const {
    return ::fabs(a - b) <= tol;
}

template<typename T>
CUDA_CALLABLE constexpr T Clamp<T>::operator()(const T &a, const T &low,
                                               const T &high) const {
    return clamp(a, low, high);
}

}// namespace vox