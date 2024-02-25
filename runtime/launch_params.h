//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/define.h"

#ifdef __CUDA_ARCH__
#if __CUDA_ARCH__ == 890
#define SM_TAG sm_89
#elif __CUDA_ARCH__ == 860
#define SM_TAG sm_86
#elif __CUDA_ARCH__ == 800
#define SM_TAG sm_80
#elif __CUDA_ARCH__ == 750
#define SM_TAG sm_75
#elif __CUDA_ARCH__ >= 700
#define SM_TAG sm_70
#elif __CUDA_ARCH__ == 620
#define SM_TAG sm_62
#elif __CUDA_ARCH__ >= 610
#define SM_TAG sm_61
#elif __CUDA_ARCH__ >= 600
#define SM_TAG sm_60
#elif __CUDA_ARCH__ == 530
#define SM_TAG sm_53
#elif __CUDA_ARCH__ >= 520
#define SM_TAG sm_52
#elif __CUDA_ARCH__ >= 500
#define SM_TAG sm_50
#elif __CUDA_ARCH__ == 370
#define SM_TAG sm_37
#elif __CUDA_ARCH__ >= 350
#define SM_TAG sm_35
#elif __CUDA_ARCH__ == 320
#define SM_TAG sm_32
#elif __CUDA_ARCH__ >= 300
#define SM_TAG sm_30
#elif __CUDA_ARCH__ >= 210
#define SM_TAG sm_21
#elif __CUDA_ARCH__ >= 200
#define SM_TAG sm_20
#else
#error "Undefined sm"
#endif
#else// __CUDA_ARCH__
#define SM_TAG sm_00
#endif

#define LAUNCH_PARAMS(launch_box) \
    typename launch_box::SM_TAG
#define LAUNCH_BOUNDS(launch_box) \
    __launch_bounds__(launch_box::sm_ptx::nt, launch_box::sm_ptx::occ)

namespace vox {
CUDA_CALLABLE constexpr int div_up(int x, int y) {
    return (x + y - 1) / y;
}

struct __attribute__((aligned(8))) cta_dim_t {
    int nt, vt;
    [[nodiscard]] int nv() const { return nt * vt; }
    [[nodiscard]] int num_ctas(int count) const {
        return div_up(count, nv());
    }
};

namespace detail {

// Due to a bug in the compiler we need to expand make_restrict() before
// branching on cta < num_ctas.
template<typename func_t, typename... args_t>
CUDA_CALLABLE_DEVICE void restrict_forward(func_t f, int tid, int cta, int num_ctas,
                                           args_t... args) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
    if (cta < num_ctas)
#endif
        f(tid, cta, args...);
}

}// namespace detail

// Generic thread cta kernel.
template<typename launch_box, typename func_t, typename... args_t>
__global__ LAUNCH_BOUNDS(launch_box) void launch_box_cta_k(func_t f, int num_ctas, args_t... args) {
    // Masking threadIdx.x by (nt - 1) may help strength reduction because the
    // compiler now knows the range of tid: (0, nt).
    typedef typename launch_box::sm_ptx params_t;
    int tid = (int)(threadIdx.x % (unsigned)params_t::nt);
    int cta = blockIdx.x;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 300
    cta += gridDim.x * blockIdx.y;
#endif

    detail::restrict_forward(f, tid, cta, num_ctas, make_restrict(args)...);
}

template<int nt_, int vt_ = 1, int vt0_ = vt_, int occ_ = 0>
struct launch_cta_t {
    enum { nt = nt_,
           vt = vt_,
           vt0 = vt0_,
           occ = occ_ };
};

#define DEF_ARCH_STRUCT(ver)                                \
    template<typename params_t, typename base_t = empty_t>  \
    struct arch_##ver : base_t {                            \
        typedef params_t sm_##ver;                          \
                                                            \
        template<typename new_base_t>                       \
        using rebind = arch_##ver<params_t, new_base_t>;    \
    };                                                      \
                                                            \
    template<int nt, int vt = 1, int vt0 = vt, int occ = 0> \
    using arch_##ver##_cta = arch_##ver<launch_cta_t<nt, vt, vt0, occ>>;

DEF_ARCH_STRUCT(20)
DEF_ARCH_STRUCT(21)
DEF_ARCH_STRUCT(30)
DEF_ARCH_STRUCT(32)
DEF_ARCH_STRUCT(35)
DEF_ARCH_STRUCT(37)
DEF_ARCH_STRUCT(50)
DEF_ARCH_STRUCT(52)
DEF_ARCH_STRUCT(53)
DEF_ARCH_STRUCT(60)
DEF_ARCH_STRUCT(61)
DEF_ARCH_STRUCT(62)
DEF_ARCH_STRUCT(70)
DEF_ARCH_STRUCT(75)
DEF_ARCH_STRUCT(80)
DEF_ARCH_STRUCT(86)
DEF_ARCH_STRUCT(89)

#undef DEF_ARCH_STRUCT

// Non-specializable launch parameters.
template<int nt, int vt, int vt0 = vt, int occ = 0>
struct launch_params_t : launch_cta_t<nt, vt, vt0, occ> {
    typedef launch_params_t sm_ptx;

    static cta_dim_t cta_dim() {
        return cta_dim_t{nt, vt};
    }

    static cta_dim_t cta_dim(int) {
        return cta_dim();
    }

    static cta_dim_t cta_dim(const Stream &stream) {
        return cta_dim();
    }

    static int nv(const Stream &stream) {
        return cta_dim().nv();
    }
};

}// namespace vox
