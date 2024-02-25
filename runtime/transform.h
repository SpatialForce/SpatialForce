//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <random>
#include <algorithm>
#include <cuda.h>
#include "launch_box.h"

namespace vox {

////////////////////////////////////////////////////////////////////////////////
// Launch a grid given a number of CTAs.

template<typename launch_box, typename func_t, typename... args_t>
void cta_launch(func_t f, int num_ctas, const Stream &stream, args_t... args) {
    auto ptx_version = stream.device().info().ptx_version;
    cta_dim_t cta = launch_box::cta_dim(ptx_version);
    dim3 grid_dim(num_ctas);
    if (ptx_version < 30 && num_ctas > 65535)
        grid_dim = dim3(256, div_up(num_ctas, 256));

    if (num_ctas)
        launch_box_cta_k<launch_box, func_t>
            <<<grid_dim, cta.nt, 0, stream.handle()>>>(f, num_ctas, args...);
}

template<int nt, int vt = 1, typename func_t, typename... args_t>
void cta_launch(func_t f, int num_ctas, const Stream &stream, args_t... args) {
    cta_launch<launch_params_t<nt, vt>>(f, num_ctas, stream, args...);
}

////////////////////////////////////////////////////////////////////////////////
// Launch a grid given a number of work-items.

template<typename launch_box, typename func_t, typename... args_t>
void cta_transform(func_t f, int count, const Stream &stream, args_t... args) {
    cta_dim_t cta = launch_box::cta_dim(stream.device().info().ptx_version);
    int num_ctas = div_up(count, cta.nv());
    cta_launch<launch_box>(f, num_ctas, stream, args...);
}

template<int nt, int vt = 1, typename func_t, typename... args_t>
void cta_transform(func_t f, int count, const Stream &stream, args_t... args) {
    cta_transform<launch_params_t<nt, vt>>(f, count, stream, args...);
}

////////////////////////////////////////////////////////////////////////////////
// Launch persistent CTAs and loop through num_ctas values.

template<typename launch_box, typename func_t, typename... args_t>
void cta_launch(func_t f, const int *num_tiles, const Stream &stream, args_t... args) {
    // Over-subscribe the device by a factor of 8.
    // This reduces the penalty if we can't schedule all the CTAs to run
    // concurrently.
    int num_ctas = 8 * occupancy<launch_box>(f, stream);

    auto k = [=] CUDA_CALLABLE_DEVICE(int tid, int cta, args_t... args) {
        int count = *num_tiles;
        while (cta < count) {
            f(tid, cta, args...);
            cta += num_ctas;
        }
    };
    cta_launch<launch_box>(k, num_ctas, stream, args...);
}

////////////////////////////////////////////////////////////////////////////////
// Ordinary transform launch. This uses the standard launch box mechanism
// so we can query its occupancy and other things.

namespace detail {

template<typename launch_t>
struct transform_f {
    template<typename func_t, typename... args_t>
    CUDA_CALLABLE_DEVICE void operator()(int tid, int cta, func_t f,
                                         size_t count, args_t... args) {

        typedef typename launch_t::sm_ptx params_t;
        enum { nt = params_t::nt,
               vt = params_t::vt,
               vt0 = params_t::vt0 };

        range_t range = get_tile(cta, nt * vt, count);

        strided_iterate<nt, vt, vt0>(
            [=](int i, int j) {
                f(range.begin + j, args...);
            },
            tid, range.count());
    }
};

}// namespace detail

template<typename launch_t, typename func_t, typename... args_t>
void transform(func_t f, size_t count, const Stream &stream, args_t... args) {
    cta_transform<launch_t>(detail::transform_f<launch_t>(), count,
                            stream, f, count, args...);
}

template<size_t nt = 128, int vt = 1, typename func_t, typename... args_t>
void transform(func_t f, size_t count, const Stream &stream, args_t... args) {
    transform<launch_params_t<nt, vt>>(f, count, stream, args...);
}

// nt controls the size of the CTA (thread block) in threads. vt is the number of values per thread (grain size).
// vt0 is the number of unconditional loads of input made in cooperative parallel kernels
// (usually set to vt for regularly parallel functions and vt - 1 for load-balancing search functions).
// occ is the occupancy argument of the __launch_bounds__ kernel decorator.
// It specifies the minimum CTAs launched concurrently on each SM. 0 is the default,
// and allows the register allocator to optimize away spills by allocating many registers to hold live state.
// Setting a specific value for this increases occupancy by limiting register usage,
// but potentially causes spills to local memory. arch_xx_cta structs support all four arguments,
// although nt is the only argument required.

}// namespace vox
