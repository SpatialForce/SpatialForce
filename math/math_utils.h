//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstddef>
#include <thrust/limits.h>
#include "core/define.h"

namespace vox {

//!
//! \brief      Returns true if \p x and \p y are similar.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//! \param[in]  eps   The tolerance.
//!
//! \tparam     T     Value type.
//!
//! \return     True if similar.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, bool>//
similar(T x, T y, T eps = thrust::numeric_limits<T>::epsilon());

//!
//! \brief      Returns the sign of the value.
//!
//! \param[in]  x     Input value.
//!
//! \tparam     T     Value type.
//!
//! \return     The sign.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
sign(T x);

//!
//! \brief      Returns the minimum value among three inputs.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//! \param[in]  z     The three value.
//!
//! \tparam     T     Value type.
//!
//! \return     The minimum value.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
min3(T x, T y, T z);

//!
//! \brief      Returns the maximum value among three inputs.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//! \param[in]  z     The three value.
//!
//! \tparam     T     Value type.
//!
//! \return     The maximum value.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
max3(T x, T y, T z);

//! Returns minimum among n-elements.
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
minn(const T *x, size_t n);

//! Returns maximum among n-elements.
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
maxn(const T *x, size_t n);

//!
//! \brief      Returns the absolute minimum value among the two inputs.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//!
//! \tparam     T     Value type.
//!
//! \return     The absolute minimum.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
absmin(T x, T y);

//!
//! \brief      Returns the absolute maximum value among the two inputs.
//!
//! \param[in]  x     The first value.
//! \param[in]  y     The second value.
//!
//! \tparam     T     Value type.
//!
//! \return     The absolute maximum.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
absmax(T x, T y);

//! Returns absolute minimum among n-elements.
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
absminn(const T *x, size_t n);

//! Returns absolute maximum among n-elements.
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
absmaxn(const T *x, size_t n);

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, size_t>//
argmin2(T x, T y);

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, size_t>//
argmax2(T x, T y);

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, size_t>//
argmin3(T x, T y, T z);

template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, size_t>//
argmax3(T x, T y, T z);

//!
//! \brief      Returns the square of \p x.
//!
//! \param[in]  x     The input.
//!
//! \tparam     T     Value type.
//!
//! \return     The squared value.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
square(T x);

//!
//! \brief      Returns the cubic of \p x.
//!
//! \param[in]  x     The input.
//!
//! \tparam     T     Value type.
//!
//! \return     The cubic of \p x.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
cubic(T x);

//!
//! \brief      Returns the clamped value.
//!
//! \param[in]  val   The value.
//! \param[in]  low   The low value.
//! \param[in]  high  The high value.
//!
//! \tparam     T     Value type.
//!
//! \return     The clamped value.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
clamp(T val, T low, T high);

//!
//! \brief      Converts degrees to radians.
//!
//! \param[in]  angleInDegrees The angle in degrees.
//!
//! \tparam     T              Value type.
//!
//! \return     Angle in radians.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
degreesToRadians(T angleInDegrees);

//!
//! \brief      Converts radians to degrees.
//!
//! \param[in]  angleInDegrees The angle in radians.
//!
//! \tparam     T              Value type.
//!
//! \return     Angle in degrees.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
radiansToDegrees(T angleInRadians);

//!
//! \brief      Computes the barycentric coordinate.
//!
//! This function computes the barycentric coordinate for given array range as
//! shown below:
//!
//! \code
//!
//! iBegin              iEnd
//! |----|-x--|----|----|
//!      i
//! t = x - i
//!
//! \endcode
//!
//! For instance, if iBegin = 4, iEnd = 8, and x = 5.4, output i will be 5 and t
//! will be 0.4.
//!
//! \param[in]  x       The input value.
//! \param[in]  iBegin  Beginning index of the range.
//! \param[in]  iEnd    End index of the range (exclusive).
//! \param[out] i       The output index between iBegin and iEnd - 2.
//! \param[out] t       The offset from \p i.
//!
//! \tparam     T       Value type.
//!
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value>//
getBarycentric(T x, ssize_t iBegin, ssize_t iEnd, ssize_t &i, T &t);

//!
//! \brief      Computes linear interpolation.
//!
//! \param[in]  f0    The first value.
//! \param[in]  f1    The second value.
//! \param[in]  t     Relative offset [0, 1] from the first value.
//!
//! \tparam     S     Input value type.
//! \tparam     T     Offset type.
//!
//! \return     The interpolated value.
//!
template<typename S, typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, S>//
lerp(const S &f0, const S &f1, T t);

//! \brief      Computes bilinear interpolation.
template<typename S, typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, S>//
bilerp(const S &f00, const S &f10, const S &f01, const S &f11, T tx, T ty);

//! \brief      Computes trilinear interpolation.
template<typename S, typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, S>//
trilerp(const S &f000, const S &f100, const S &f010, const S &f110,
        const S &f001, const S &f101, const S &f011, const S &f111, T tx, T ty,
        T tz);

//! \brief      Computes Catmull-Rom interpolation.
template<typename S, typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, S>//
catmullRom(const S &f0, const S &f1, const S &f2, const S &f3, T t);

//! \brief      Computes monotonic Catmull-Rom interpolation.
template<typename T>
CUDA_CALLABLE std::enable_if_t<std::is_arithmetic<T>::value, T>//
monotonicCatmullRom(const T &f0, const T &f1, const T &f2, const T &f3, T t);

//---------------------------------------------------------------------------------------------------------------
template<typename T>
inline CUDA_CALLABLE T atomic_add(T *buf, T value) {
#if !defined(__CUDA_ARCH__)
    T old = buf[0];
    buf[0] += value;
    return old;
#else
    return atomicAdd(buf, value);
#endif
}

// emulate atomic float max
inline CUDA_CALLABLE float atomic_max(float *address, float val) {
#if defined(__CUDA_ARCH__)
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;

    while (val > __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }

    return __int_as_float(old);

#else
    float old = *address;
    *address = max(old, val);
    return old;
#endif
}

// emulate atomic float min/max with atomicCAS()
inline CUDA_CALLABLE float atomic_min(float *address, float val) {
#if defined(__CUDA_ARCH__)
    int *address_as_int = (int *)address;
    int old = *address_as_int, assumed;

    while (val < __int_as_float(old)) {
        assumed = old;
        old = atomicCAS(address_as_int, assumed,
                        __float_as_int(val));
    }

    return __int_as_float(old);

#else
    float old = *address;
    *address = min(old, val);
    return old;
#endif
}

inline CUDA_CALLABLE int atomic_max(int *address, int val) {
#if defined(__CUDA_ARCH__)
    return atomicMax(address, val);

#else
    int old = *address;
    *address = max(old, val);
    return old;
#endif
}

// atomic int min
inline CUDA_CALLABLE int atomic_min(int *address, int val) {
#if defined(__CUDA_ARCH__)
    return atomicMin(address, val);

#else
    int old = *address;
    *address = min(old, val);
    return old;
#endif
}

}// namespace vox

#include "math_utils-inl.h"