//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "fixed_array.h"

namespace vox {

template<typename Type, size_t Length>
struct vec_t {
    Type c[Length] = {};

    inline vec_t() = default;

    inline CUDA_CALLABLE constexpr vec_t(Type s) {
        for (unsigned i = 0; i < Length; ++i) {
            c[i] = s;
        }
    }

    template<typename OtherType>
    inline explicit CUDA_CALLABLE vec_t(const vec_t<OtherType, Length> &other) {
        for (unsigned i = 0; i < Length; ++i) {
            c[i] = other[i];
        }
    }

    inline CUDA_CALLABLE constexpr vec_t(Type x, Type y) {
        assert(Length == 2);
        c[0] = x;
        c[1] = y;
    }

    inline CUDA_CALLABLE constexpr vec_t(Type x, Type y, Type z) {
        assert(Length == 3);
        c[0] = x;
        c[1] = y;
        c[2] = z;
    }

    inline CUDA_CALLABLE constexpr vec_t(Type x, Type y, Type z, Type w) {
        assert(Length == 4);
        c[0] = x;
        c[1] = y;
        c[2] = z;
        c[3] = w;
    }

    inline CUDA_CALLABLE constexpr vec_t(const fixed_array_t<Type, Length> &l) {
        for (unsigned i = 0; i < Length; ++i) {
            c[i] = l[i];
        }
    }

    // special screw vector constructor for spatial_vectors:
    inline CUDA_CALLABLE constexpr vec_t(vec_t<Type, 3> w, vec_t<Type, 3> v) {
        c[0] = w[0];
        c[1] = w[1];
        c[2] = w[2];
        c[3] = v[0];
        c[4] = v[1];
        c[5] = v[2];
    }

    inline CUDA_CALLABLE constexpr Type operator[](int index) const {
        assert(index < Length);
        return c[index];
    }

    inline CUDA_CALLABLE constexpr Type &operator[](int index) {
        assert(index < Length);
        return c[index];
    }

    CUDA_CALLABLE inline vec_t operator/=(const Type &h);

    CUDA_CALLABLE inline vec_t operator*=(const vec_t &h);

    CUDA_CALLABLE inline vec_t operator*=(const Type &h);
};

using vec2b = vec_t<int8_t, 2>;
using vec3b = vec_t<int8_t, 3>;
using vec4b = vec_t<int8_t, 4>;
using vec2ub = vec_t<uint8_t, 2>;
using vec3ub = vec_t<uint8_t, 3>;
using vec4ub = vec_t<uint8_t, 4>;

using vec2s = vec_t<int16_t, 2>;
using vec3s = vec_t<int16_t, 3>;
using vec4s = vec_t<int16_t, 4>;
using vec2us = vec_t<uint16_t, 2>;
using vec3us = vec_t<uint16_t, 3>;
using vec4us = vec_t<uint16_t, 4>;

using vec2i = vec_t<int32_t, 2>;
using vec3i = vec_t<int32_t, 3>;
using vec4i = vec_t<int32_t, 4>;
using vec2ui = vec_t<uint32_t, 2>;
using vec3ui = vec_t<uint32_t, 3>;
using vec4ui = vec_t<uint32_t, 4>;

using vec2l = vec_t<int64_t, 2>;
using vec3l = vec_t<int64_t, 3>;
using vec4l = vec_t<int64_t, 4>;
using vec2ul = vec_t<uint64_t, 2>;
using vec3ul = vec_t<uint64_t, 3>;
using vec4ul = vec_t<uint64_t, 4>;

using vec2 = vec_t<float, 2>;
using vec3 = vec_t<float, 3>;
using vec4 = vec_t<float, 4>;

using vec2f = vec_t<float, 2>;
using vec3f = vec_t<float, 3>;
using vec4f = vec_t<float, 4>;

using vec2d = vec_t<double, 2>;
using vec3d = vec_t<double, 3>;
using vec4d = vec_t<double, 4>;

//--------------
// vec<Length, Type> methods

// Should these accept const references as arguments? It's all
// inlined so maybe it doesn't matter? Even if it does, it
// probably depends on the Length of the vector...

// negation:
template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> operator-(vec_t<Type, Length> a) {
    // NB: this constructor will initialize all ret's components to 0, which is
    // unnecessary...
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = -a[i];
    }

    // Wonder if this does a load of copying when it returns... hopefully not as it's inlined?
    return ret;
}

template<typename Type, size_t Length>
CUDA_CALLABLE inline vec_t<Type, Length> pos(const vec_t<Type, Length> &x) {
    return x;
}

template<typename Type, size_t Length>
CUDA_CALLABLE inline vec_t<Type, Length> neg(const vec_t<Type, Length> &x) {
    return -x;
}

template<typename Type>
CUDA_CALLABLE inline vec_t<Type, 3> neg(const vec_t<Type, 3> &x) {
    return vec_t<Type, 3>(-x.c[0], -x.c[1], -x.c[2]);
}

template<typename Type>
CUDA_CALLABLE inline vec_t<Type, 2> neg(const vec_t<Type, 2> &x) {
    return vec_t<Type, 2>(-x.c[0], -x.c[1]);
}

// equality:
template<typename Type, size_t Length>
inline CUDA_CALLABLE bool operator==(const vec_t<Type, Length> &a, const vec_t<Type, Length> &b) {
    for (unsigned i = 0; i < Length; ++i) {
        if (a[i] != b[i]) {
            return false;
        }
    }
    return true;
}

// scalar multiplication:
template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> mul(vec_t<Type, Length> a, Type s) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] * s;
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 3> mul(vec_t<Type, 3> a, Type s) {
    return vec_t<Type, 3>(a.c[0] * s, a.c[1] * s, a.c[2] * s);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 2> mul(vec_t<Type, 2> a, Type s) {
    return vec_t<Type, 2>(a.c[0] * s, a.c[1] * s);
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> mul(Type s, vec_t<Type, Length> a) {
    return mul(a, s);
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> operator*(Type s, vec_t<Type, Length> a) {
    return mul(a, s);
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> operator*(vec_t<Type, Length> a, Type s) {
    return mul(a, s);
}

template<typename Type, size_t Length>
CUDA_CALLABLE inline vec_t<Type, Length> vec_t<Type, Length>::operator*=(const vec_t &h) {
    *this = mul(*this, h);
    return *this;
}

template<typename Type, size_t Length>
CUDA_CALLABLE inline vec_t<Type, Length> vec_t<Type, Length>::operator*=(const Type &h) {
    *this = mul(*this, h);
    return *this;
}

// component wise multiplication:
template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> cw_mul(vec_t<Type, Length> a, vec_t<Type, Length> b) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] * b[i];
    }
    return ret;
}

// division
template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> div(vec_t<Type, Length> a, Type s) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] / s;
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 3> div(vec_t<Type, 3> a, Type s) {
    return vec_t<Type, 3>(a.c[0] / s, a.c[1] / s, a.c[2] / s);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 2> div(vec_t<Type, 2> a, Type s) {
    return vec_t<Type, 2>(a.c[0] / s, a.c[1] / s);
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> operator/(vec_t<Type, Length> a, Type s) {
    return div(a, s);
}

template<typename Type, size_t Length>
CUDA_CALLABLE inline vec_t<Type, Length> vec_t<Type, Length>::operator/=(const Type &h) {
    *this = div(*this, h);
    return *this;
}

// component wise division
template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> cw_div(vec_t<Type, Length> a, vec_t<Type, Length> b) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] / b[i];
    }
    return ret;
}

// addition
template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> add(vec_t<Type, Length> a, vec_t<Type, Length> b) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] + b[i];
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 2> add(vec_t<Type, 2> a, vec_t<Type, 2> b) {
    return vec_t<Type, 2>(a.c[0] + b.c[0], a.c[1] + b.c[1]);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 3> add(vec_t<Type, 3> a, vec_t<Type, 3> b) {
    return vec_t<Type, 3>(a.c[0] + b.c[0], a.c[1] + b.c[1], a.c[2] + b.c[2]);
}

// subtraction
template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> sub(vec_t<Type, Length> a, vec_t<Type, Length> b) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = Type(a[i] - b[i]);
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 2> sub(vec_t<Type, 2> a, vec_t<Type, 2> b) {
    return vec_t<Type, 2>(a.c[0] - b.c[0], a.c[1] - b.c[1]);
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 3> sub(vec_t<Type, 3> a, vec_t<Type, 3> b) {
    return vec_t<Type, 3>(a.c[0] - b.c[0], a.c[1] - b.c[1], a.c[2] - b.c[2]);
}

// dot product:
template<typename Type, size_t Length>
inline CUDA_CALLABLE Type dot(vec_t<Type, Length> a, vec_t<Type, Length> b) {
    Type ret(0);
    for (unsigned i = 0; i < Length; ++i) {
        ret += a[i] * b[i];
    }
    return ret;
}

template<typename Type>
inline CUDA_CALLABLE Type dot(vec_t<Type, 2> a, vec_t<Type, 2> b) {
    return a.c[0] * b.c[0] + a.c[1] * b.c[1];
}

template<typename Type>
inline CUDA_CALLABLE Type dot(vec_t<Type, 3> a, vec_t<Type, 3> b) {
    return a.c[0] * b.c[0] + a.c[1] * b.c[1] + a.c[2] * b.c[2];
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE Type tensordot(vec_t<Type, Length> a, vec_t<Type, Length> b) {
    // corresponds to `np.tensordot()` with all axes being contracted
    return dot(a, b);
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE Type &index(const vec_t<Type, Length> &a, int idx) {
#ifndef NDEBUG
    if (idx < 0 || idx >= Length) {
        printf("vec index %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    return a[idx];
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE void indexset(vec_t<Type, Length> &v, int idx, Type value) {
#ifndef NDEBUG
    if (idx < 0 || idx >= Length) {
        printf("vec store %d out of bounds at %s %d\n", idx, __FILE__, __LINE__);
        assert(0);
    }
#endif

    v[idx] = value;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE Type length(vec_t<Type, Length> a) {
    return sqrt(dot(a, a));
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE Type length_sq(vec_t<Type, Length> a) {
    return dot(a, a);
}

template<typename Type>
inline CUDA_CALLABLE Type length(vec_t<Type, 2> a) {
    return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
}

template<typename Type>
inline CUDA_CALLABLE Type length(vec_t<Type, 3> a) {
    return sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> normalize(vec_t<Type, Length> a) {
    Type l = length(a);
    if (l > Type(kEps))
        return div(a, l);
    else
        return vec_t<Type, Length>();
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 2> normalize(vec_t<Type, 2> a) {
    Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1]);
    if (l > Type(kEps))
        return vec_t<Type, 2>(a.c[0] / l, a.c[1] / l);
    else
        return vec_t<Type, 2>();
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 3> normalize(vec_t<Type, 3> a) {
    Type l = sqrt(a.c[0] * a.c[0] + a.c[1] * a.c[1] + a.c[2] * a.c[2]);
    if (l > Type(kEps))
        return vec_t<Type, 3>(a.c[0] / l, a.c[1] / l, a.c[2] / l);
    else
        return vec_t<Type, 3>();
}

template<typename Type>
inline CUDA_CALLABLE vec_t<Type, 3> cross(vec_t<Type, 3> a, vec_t<Type, 3> b) {
    return {Type(a[1] * b[2] - a[2] * b[1]), Type(a[2] * b[0] - a[0] * b[2]), Type(a[0] * b[1] - a[1] * b[0])};
}

template<typename Type, size_t Length>
inline bool CUDA_CALLABLE isfinite(vec_t<Type, Length> x) {
    for (unsigned i = 0; i < Length; ++i) {
        if (!isfinite(x[i])) {
            return false;
        }
    }
    return true;
}

// These two functions seem to compile very slowly
template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> min(vec_t<Type, Length> a, vec_t<Type, Length> b) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] < b[i] ? a[i] : b[i];
    }
    return ret;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> max(vec_t<Type, Length> a, vec_t<Type, Length> b) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = a[i] > b[i] ? a[i] : b[i];
    }
    return ret;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE Type min(vec_t<Type, Length> v) {
    Type ret = v[0];
    for (unsigned i = 1; i < Length; ++i) {
        if (v[i] < ret) ret = v[i];
    }
    return ret;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE Type max(vec_t<Type, Length> v) {
    Type ret = v[0];
    for (unsigned i = 1; i < Length; ++i) {
        if (v[i] > ret) ret = v[i];
    }
    return ret;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE unsigned argmin(vec_t<Type, Length> v) {
    unsigned ret = 0;
    for (unsigned i = 1; i < Length; ++i) {
        if (v[i] < v[ret]) ret = i;
    }
    return ret;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE unsigned argmax(vec_t<Type, Length> v) {
    unsigned ret = 0;
    for (unsigned i = 1; i < Length; ++i) {
        if (v[i] > v[ret]) ret = i;
    }
    return ret;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE void expect_near(const vec_t<Type, Length> &actual,
                                      const vec_t<Type, Length> &expected,
                                      const Type &tolerance) {
    const Type diff(0);
    for (size_t i = 0; i < Length; ++i) {
        diff = max(diff, abs(actual[i] - expected[i]));
    }
    if (diff > tolerance) {
        printf("Error, expect_near() failed with tolerance ");
        print(tolerance);
        printf("\t Expected: ");
        print(expected);
        printf("\t Actual: ");
        print(actual);
    }
}

// Do I need to specialize these for different lengths?
template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> atomic_add(vec_t<Type, Length> *addr, vec_t<Type, Length> value) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_add(&(addr->c[i]), value[i]);
    }

    return ret;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> atomic_min(vec_t<Type, Length> *addr, vec_t<Type, Length> value) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_min(&(addr->c[i]), value[i]);
    }

    return ret;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE vec_t<Type, Length> atomic_max(vec_t<Type, Length> *addr, vec_t<Type, Length> value) {
    vec_t<Type, Length> ret;
    for (unsigned i = 0; i < Length; ++i) {
        ret[i] = atomic_max(&(addr->c[i]), value[i]);
    }

    return ret;
}

// ok, the original implementation of this didn't take the absolute values.
// I wouldn't consider this expected behavior. It looks like it's only
// being used for bounding boxes at the moment, where this doesn't matter,
// but you often use it for ray tracing where it does. Not sure if the
// fabs() incurs a performance hit...
template<typename Type, size_t Length>
CUDA_CALLABLE inline int longest_axis(const vec_t<Type, Length> &v) {
    Type lmax = abs(v[0]);
    int ret(0);
    for (unsigned i = 1; i < Length; ++i) {
        Type l = abs(v[i]);
        if (l > lmax) {
            ret = i;
            lmax = l;
        }
    }
    return ret;
}

template<typename Type, size_t Length>
CUDA_CALLABLE inline vec_t<Type, Length> lerp(const vec_t<Type, Length> &a, const vec_t<Type, Length> &b, Type t) {
    return a * (Type(1) - t) + b * t;
}

template<typename Type, size_t Length>
inline CUDA_CALLABLE void print(vec_t<Type, Length> v) {
    for (unsigned i = 0; i < Length; ++i) {
        printf("%g ", float(v[i]));
    }
    printf("\n");
}

inline CUDA_CALLABLE void expect_near(const vec3 &actual, const vec3 &expected, const float &tolerance) {
    const float diff =
        fmax(fmax(abs(actual[0] - expected[0]), abs(actual[1] - expected[1])), abs(actual[2] - expected[2]));
    if (diff > tolerance) {
        printf("Error, expect_near() failed with tolerance ");
        printf("%g\n", tolerance);
        printf("\t Expected: ");
        print(expected);
        printf("\t Actual: ");
        print(actual);
    }
}

}// namespace vox