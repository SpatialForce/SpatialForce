//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "constants.h"
#include <algorithm>
#include <cmath>
#include <cassert>
#include <thrust/limits.h>

namespace vox {

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, bool>//
similar(T x, T y, T eps) {
    return (::abs(x - y) <= eps);
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
sign(T x) {
    if (x >= 0) {
        return 1;
    } else {
        return -1;
    }
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
min3(T x, T y, T z) {
    return ::min(::min(x, y), z);
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
max3(T x, T y, T z) {
    return ::max(::max(x, y), z);
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
minn(const T *x, size_t n) {
    T m = x[0];
    for (size_t i = 1; i < n; i++) {
        m = ::min(m, x[i]);
    }
    return m;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
maxn(const T *x, size_t n) {
    T m = x[0];
    for (size_t i = 1; i < n; i++) {
        m = ::max(m, x[i]);
    }
    return m;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
absmin(T x, T y) {
    return (x * x < y * y) ? x : y;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
absmax(T x, T y) {
    return (x * x > y * y) ? x : y;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
absminn(const T *x, size_t n) {
    T m = x[0];
    for (size_t i = 1; i < n; i++) {
        m = absmin(m, x[i]);
    }
    return m;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
absmaxn(const T *x, size_t n) {
    T m = x[0];
    for (size_t i = 1; i < n; i++) {
        m = absmax(m, x[i]);
    }
    return m;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, size_t>//
argmin2(T x, T y) {
    return (x < y) ? 0 : 1;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, size_t>//
argmax2(T x, T y) {
    return (x > y) ? 0 : 1;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, size_t>//
argmin3(T x, T y, T z) {
    if (x < y) {
        return (x < z) ? 0 : 2;
    } else {
        return (y < z) ? 1 : 2;
    }
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, size_t>//
argmax3(T x, T y, T z) {
    if (x > y) {
        return (x > z) ? 0 : 2;
    } else {
        return (y > z) ? 1 : 2;
    }
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
square(T x) {
    return x * x;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
cubic(T x) {
    return x * x * x;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
clamp(T val, T low, T high) {
    if (val < low) {
        return low;
    } else if (val > high) {
        return high;
    } else {
        return val;
    }
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
degreesToRadians(T angleInDegrees) {
    return angleInDegrees * pi<T>() / 180;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
radiansToDegrees(T angleInRadians) {
    return angleInRadians * 180 / pi<T>();
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value>//
getBarycentric(T x, ssize_t iBegin, ssize_t iEnd, ssize_t &i, T &f) {
    assert(iEnd > iBegin);

    T s = std::floor(x);
    i = static_cast<ssize_t>(s);
    ssize_t size = iEnd - iBegin;

    if (size == 1 || i < 0) {
        i = iBegin;
        f = 0;
    } else if (i > iEnd - 2) {
        i = iEnd - 2;
        f = 1;
    } else {
        f = static_cast<T>(x - s);
    }
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value>//
getBarycentric(T x, ssize_t iEnd, ssize_t &i, T &f) {
    assert(iEnd > 0);

    T s = std::floor(x);
    i = static_cast<ssize_t>(s);
    f = x - s;

    if (iEnd == 1 || i < 0) {
        i = 0;
        f = 0;
    } else if (i > iEnd - 2) {
        i = iEnd - 2;
        f = 1;
    }
}

template<typename S, typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, S>//
lerp(const S &value0, const S &value1, T f) {
    return (1 - f) * value0 + f * value1;
}

template<typename S, typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, S>//
bilerp(const S &f00, const S &f10, const S &f01, const S &f11, T tx, T ty) {
    return lerp(lerp(f00, f10, tx), lerp(f01, f11, tx), ty);
}

template<typename S, typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, S>//
trilerp(const S &f000, const S &f100, const S &f010, const S &f110,
        const S &f001, const S &f101, const S &f011, const S &f111, T tx, T ty,
        T fz) {
    return lerp(bilerp(f000, f100, f010, f110, tx, ty),
                bilerp(f001, f101, f011, f111, tx, ty), fz);
}

template<typename S, typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, S>//
catmullRom(const S &f0, const S &f1, const S &f2, const S &f3, T f) {
    S d1 = (f2 - f0) / 2;
    S d2 = (f3 - f1) / 2;
    S D1 = f2 - f1;

    S a3 = d1 + d2 - 2 * D1;
    S a2 = 3 * D1 - 2 * d1 - d2;
    S a1 = d1;
    S a0 = f1;

    return a3 * cubic(f) + a2 * square(f) + a1 * f + a0;
}

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
monotonicCatmullRom(const T &f0, const T &f1, const T &f2, const T &f3, T f) {
    T d1 = (f2 - f0) / 2;
    T d2 = (f3 - f1) / 2;
    T D1 = f2 - f1;

    if (std::fabs(D1) < kEpsilonD) {
        d1 = d2 = 0;
    }

    if (sign(D1) != sign(d1)) {
        d1 = 0;
    }

    if (sign(D1) != sign(d2)) {
        d2 = 0;
    }

    T a3 = d1 + d2 - 2 * D1;
    T a2 = 3 * D1 - 2 * d1 - d2;
    T a1 = d1;
    T a0 = f1;

    return a3 * cubic(f) + a2 * square(f) + a1 * f + a0;
}

}// namespace vox