//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "matrix.h"

namespace vox {
template<typename T>
class SpatialVector {
public:
    Vector<T, 3> w;
    Vector<T, 3> v;

    CUDA_CALLABLE T dot(const SpatialVector &b) {
        return w.dot(b.w) + v.dot(b.v);
    }

    CUDA_CALLABLE SpatialVector<T> cross(const SpatialVector<T> &b) {
        auto ww = w.cross(b.w);
        auto vv = v.cross(b.w) + w.cross(b.v);
        return {ww, vv};
    }

    CUDA_CALLABLE SpatialVector<T> cross_dual(const SpatialVector<T> &b) {
        auto ww = w.cross(b.w) + v.cross(b.v);
        auto vv = w.cross(b.v);
        return {ww, vv};
    }
};

using SpatialVectorF = SpatialVector<float>;
using SpatialVectorD = SpatialVector<double>;

}// namespace vox