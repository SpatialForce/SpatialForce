//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "quaternion.h"

namespace vox {
template<typename Type>
struct Transform {
    Vector<Type, 3> p;
    Quaternion<Type> q;

    CUDA_CALLABLE inline Transform(Vector<Type, 3> p = Vector<Type, 3>(), Quaternion<Type> q = Quaternion<Type>()) : p(p), q(q) {}

    CUDA_CALLABLE inline Type operator[](int index) const {
        assert(index < 7);

        return p.c[index];
    }

    CUDA_CALLABLE inline Type &operator[](int index) {
        assert(index < 7);

        return p.c[index];
    }

    CUDA_CALLABLE inline Transform<Type> mul(const Transform<Type> &b) {
        return {q * b.p + p, q * b.q};
    }

    CUDA_CALLABLE inline Transform<Type> inverse() {
        auto q_inv = q.inverse();
        return transform_t<Type>(-q_inv * p, q_inv);
    }

    CUDA_CALLABLE inline Vector<Type, 3> transform_vector(const Vector<Type, 3> &x) {
        return q * x;
    }

    CUDA_CALLABLE inline Vector<Type, 3> transform_point(const Vector<Type, 3> &x) {
        return p + q * x;
    }
};

template<typename Type>
inline CUDA_CALLABLE bool operator==(const Transform<Type> &a, const Transform<Type> &b) {
    return a.p == b.p && a.q == b.q;
}

// not totally sure why you'd want to do this seeing as adding/subtracting two rotation
// quats doesn't seem to do anything meaningful
template<typename Type>
CUDA_CALLABLE inline Transform<Type> add(const Transform<Type> &a, const Transform<Type> &b) {
    return {a.p + b.p, a.q + b.q};
}

template<typename Type>
CUDA_CALLABLE inline Transform<Type> sub(const Transform<Type> &a, const Transform<Type> &b) {
    return {a.p - b.p, a.q - b.q};
}

// also not sure why you'd want to do this seeing as the quat would end up unnormalized
template<typename Type>
CUDA_CALLABLE inline Transform<Type> mul(const Transform<Type> &a, Type s) {
    return {a.p * s, a.q * s};
}

template<typename Type>
CUDA_CALLABLE inline Transform<Type> mul(Type s, const Transform<Type> &a) {
    return mul(a, s);
}

template<typename Type>
CUDA_CALLABLE inline Transform<Type> mul(const Transform<Type> &a, const Transform<Type> &b) {
    return a.mul(b);
}

template<typename Type>
CUDA_CALLABLE inline Transform<Type> operator*(const Transform<Type> &a, Type s) {
    return mul(a, s);
}

template<typename Type>
CUDA_CALLABLE inline Transform<Type> operator*(Type s, const Transform<Type> &a) {
    return mul(a, s);
}

template<typename Type>
inline CUDA_CALLABLE Type extract(const Transform<Type> &t, int i) {
    return t[i];
}

template<typename Type>
CUDA_CALLABLE inline Transform<Type> lerp(const Transform<Type> &a, const Transform<Type> &b, Type t) {
    return a * (Type(1) - t) + b * t;
}

using TransformF = Transform<float>;
using TransformD = Transform<double>;

}// namespace vox