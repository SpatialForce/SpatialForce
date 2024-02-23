//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <functional>
#include <limits>
#include "core/define.h"

namespace vox {

//! No-op operator.
template<typename T>
struct NoOp {
    CUDA_CALLABLE constexpr T operator()(const T &a) const;
};

//! Type casting operator.
template<typename T, typename U>
struct TypeCast {
    CUDA_CALLABLE constexpr U operator()(const T &a) const;
};

//! Performs std::ceil.
template<typename T>
struct Ceil {
    CUDA_CALLABLE constexpr T operator()(const T &a) const;
};

//! Performs std::floor.
template<typename T>
struct Floor {
    CUDA_CALLABLE constexpr T operator()(const T &a) const;
};

//! Square operator (a * a).
template<typename T>
struct Square {
    CUDA_CALLABLE constexpr T operator()(const T &a) const;
};

//! Reverse minus operator.
template<typename T>
struct RMinus {
    CUDA_CALLABLE constexpr T operator()(const T &a, const T &b) const;
};

//! Reverse divides operator.
template<typename T>
struct RDivides {
    CUDA_CALLABLE constexpr T operator()(const T &a, const T &b) const;
};

//! Add-and-assign operator (+=).
template<typename T>
struct IAdd {
    CUDA_CALLABLE void operator()(T &a, const T &b) const;
};

//! Subtract-and-assign operator (-=).
template<typename T>
struct ISub {
    CUDA_CALLABLE void operator()(T &a, const T &b) const;
};

//! Multiply-and-assign operator (*=).
template<typename T>
struct IMul {
    CUDA_CALLABLE void operator()(T &a, const T &b) const;
};

//! Divide-and-assign operator (/=).
template<typename T>
struct IDiv {
    CUDA_CALLABLE void operator()(T &a, const T &b) const;
};

//! Takes minimum value.
template<typename T>
struct Min {
    CUDA_CALLABLE constexpr T operator()(const T &a, const T &b) const;
};

//! Takes maximum value.
template<typename T>
struct Max {
    CUDA_CALLABLE constexpr T operator()(const T &a, const T &b) const;
};

//! Takes absolute minimum value.
template<typename T>
struct AbsMin {
    CUDA_CALLABLE constexpr T operator()(const T &a, const T &b) const;
};

//! Takes absolute maximum value.
template<typename T>
struct AbsMax {
    CUDA_CALLABLE constexpr T operator()(const T &a, const T &b) const;
};

//! True if similar
template<typename T>
struct SimilarTo {
    double tol;
    CUDA_CALLABLE constexpr SimilarTo(double tol_ = std::numeric_limits<double>::epsilon()) : tol(tol_) {}
    CUDA_CALLABLE constexpr bool operator()(const T &a, const T &b) const;
};

//! Clamps the input value with low/high.
template<typename T>
struct Clamp {
    CUDA_CALLABLE constexpr T operator()(const T &a, const T &low, const T &high) const;
};

}// namespace vox

#include "functors-inl.h"
