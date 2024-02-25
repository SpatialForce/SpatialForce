//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cmath>

#if !defined(__CUDACC__)
#define CUDA_CALLABLE
#define CUDA_CALLABLE_DEVICE
#else
#define CUDA_CALLABLE __host__ __device__
#define CUDA_CALLABLE_DEVICE __device__
#endif

// MARK: Debug mode
#if defined(NDEBUG)
#define ASSERT(x)
#else
#include <cassert>
#define ASSERT(x) assert(x)
#endif

// MARK: C++ exceptions
#ifdef __cplusplus
#include <stdexcept>
#define THROW_INVALID_ARG_IF(expression)          \
    if (expression) {                             \
        throw std::invalid_argument(#expression); \
    }
#define THROW_INVALID_ARG_WITH_MESSAGE_IF(expression, message) \
    if (expression) {                                          \
        throw std::invalid_argument(message);                  \
    }
#endif

enum { warp_size = 32 };

struct empty_t {};

template<typename... base_v>
struct inherit_t;

template<typename base_t, typename... base_v>
struct inherit_t<base_t, base_v...> : base_t::template rebind<inherit_t<base_v...>> {};

template<typename base_t>
struct inherit_t<base_t> : base_t {};

////////////////////////////////////////////////////////////////////////////////
// Conditional typedefs.

// Typedef type_a if type_a is not empty_t.
// Otherwise typedef type_b.
template<typename type_a, typename type_b>
struct conditional_typedef_t {
    typedef typename std::conditional<
        !std::is_same<type_a, empty_t>::value,
        type_a,
        type_b>::type type_t;
};

////////////////////////////////////////////////////////////////////////////////
// Code to treat __restrict__ as a CV qualifier.

// XXX: Results in a bug w/ MSC.
// Add __restrict__ only to pointers.
template<typename arg_t>
struct add_restrict {
    typedef arg_t type;
};
template<typename arg_t>
struct add_restrict<arg_t *> {
    typedef arg_t *__restrict__ type;
};

template<typename arg_t>
CUDA_CALLABLE typename add_restrict<arg_t>::type make_restrict(arg_t x) {
    typename add_restrict<arg_t>::type y = x;
    return y;
}

////////////////////////////////////////////////////////////////////////////////
// Template unrolled looping construct.

template<int i, int count, bool valid = (i < count)>
struct iterate_t {
#pragma nv_exec_check_disable
    template<typename func_t>
    CUDA_CALLABLE static void eval(func_t f) {
        f(i);
        iterate_t<i + 1, count>::eval(f);
    }
};
template<int i, int count>
struct iterate_t<i, count, false> {
    template<typename func_t>
    CUDA_CALLABLE static void eval(func_t f) {}
};
template<int begin, int end, typename func_t>
CUDA_CALLABLE void iterate(func_t f) {
    iterate_t<begin, end>::eval(f);
}
template<int count, typename func_t>
CUDA_CALLABLE void iterate(func_t f) {
    iterate<0, count>(f);
}

template<int count, typename type_t>
CUDA_CALLABLE type_t reduce(const type_t (&x)[count]) {
    type_t y;
    iterate<count>([&](int i) { y = i ? x[i] + y : x[i]; });
    return y;
}

template<int count, typename type_t>
CUDA_CALLABLE void fill(type_t (&x)[count], type_t val) {
    iterate<count>([&](int i) { x[i] = val; });
}

#ifdef __CUDACC__

// Invoke unconditionally.
template<int nt, int vt, typename func_t>
CUDA_CALLABLE_DEVICE void strided_iterate(func_t f, int tid) {
    iterate<vt>([=](int i) { f(i, nt * i + tid); });
}

// Check range.
template<int nt, int vt, int vt0 = vt, typename func_t>
CUDA_CALLABLE_DEVICE void strided_iterate(func_t f, int tid, int count) {
    // Unroll the first vt0 elements of each thread.
    if (vt0 > 1 && count >= nt * vt0) {
        strided_iterate<nt, vt0>(f, tid);// No checking
    } else {
        iterate<vt0>([=](int i) {
            int j = nt * i + tid;
            if (j < count) f(i, j);
        });
    }

    iterate<vt0, vt>([=](int i) {
        int j = nt * i + tid;
        if (j < count) f(i, j);
    });
}
template<int vt, typename func_t>
CUDA_CALLABLE_DEVICE void thread_iterate(func_t f, int tid) {
    iterate<vt>([=](int i) { f(i, vt * tid + i); });
}

#endif// ifdef __CUDACC__

struct __attribute__((aligned(8))) range_t {
    int begin, end;
    CUDA_CALLABLE int size() const { return end - begin; }
    CUDA_CALLABLE int count() const { return size(); }
    CUDA_CALLABLE bool valid() const { return end > begin; }
};

CUDA_CALLABLE inline range_t get_tile(int cta, int nv, int count) {
    return range_t{nv * cta, (count < nv * (cta + 1)) ? count : nv * (cta + 1)};
}