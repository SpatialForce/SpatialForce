//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "launch_params.h"

namespace vox {

// Specializable launch parameters.
struct launch_box_default_t {
    typedef launch_cta_t<0, 0, 0> sm_00;
    typedef empty_t sm_20, sm_21, sm_30, sm_32, sm_35, sm_37, sm_50, sm_52, sm_53,
        sm_60, sm_61, sm_62, sm_70, sm_75, sm_80, sm_86, sm_89;

    template<typename new_base_t>
    using rebind = launch_box_default_t;
};

template<typename... params_v>
struct launch_box_t : inherit_t<params_v..., launch_box_default_t> {
    typedef inherit_t<params_v..., launch_box_default_t> base_t;

    typedef typename conditional_typedef_t<
        typename base_t::sm_20, typename base_t::sm_00>::type_t sm_20;

#define INHERIT_LAUNCH_PARAMS(new_ver, old_ver) \
    typedef typename conditional_typedef_t<     \
        typename base_t::sm_##new_ver, sm_##old_ver>::type_t sm_##new_ver;

    INHERIT_LAUNCH_PARAMS(21, 20)
    INHERIT_LAUNCH_PARAMS(30, 21)
    INHERIT_LAUNCH_PARAMS(32, 30)
    INHERIT_LAUNCH_PARAMS(35, 30)
    INHERIT_LAUNCH_PARAMS(37, 35)
    INHERIT_LAUNCH_PARAMS(50, 35)
    INHERIT_LAUNCH_PARAMS(52, 50)
    INHERIT_LAUNCH_PARAMS(53, 50)
    INHERIT_LAUNCH_PARAMS(60, 53)
    INHERIT_LAUNCH_PARAMS(61, 60)
    INHERIT_LAUNCH_PARAMS(62, 60)
    INHERIT_LAUNCH_PARAMS(70, 62)
    INHERIT_LAUNCH_PARAMS(75, 70)
    INHERIT_LAUNCH_PARAMS(80, 75)
    INHERIT_LAUNCH_PARAMS(86, 80)
    INHERIT_LAUNCH_PARAMS(89, 86)

    // Overwrite the params defined for sm_00 so that the host-side compiler
    // has all expected symbols available to it.
    typedef sm_89 sm_00;
    typedef LAUNCH_PARAMS(launch_box_t) sm_ptx;

    static cta_dim_t cta_dim(int ptx_version) {
        // Ptx version from cudaFuncGetAttributes.
        if (ptx_version == 86)
            return cta_dim_t{sm_86::nt, sm_86::vt};
        else if (ptx_version == 80)
            return cta_dim_t{sm_80::nt, sm_80::vt};
        else if (ptx_version == 75)
            return cta_dim_t{sm_75::nt, sm_75::vt};
        else if (ptx_version >= 70)
            return cta_dim_t{sm_70::nt, sm_70::vt};
        else if (ptx_version == 62)
            return cta_dim_t{sm_62::nt, sm_62::vt};
        else if (ptx_version >= 61)
            return cta_dim_t{sm_61::nt, sm_61::vt};
        else if (ptx_version >= 60)
            return cta_dim_t{sm_60::nt, sm_60::vt};
        else if (ptx_version == 53)
            return cta_dim_t{sm_53::nt, sm_53::vt};
        else if (ptx_version >= 52)
            return cta_dim_t{sm_52::nt, sm_52::vt};
        else if (ptx_version >= 50)
            return cta_dim_t{sm_50::nt, sm_50::vt};
        else if (ptx_version == 37)
            return cta_dim_t{sm_37::nt, sm_37::vt};
        else if (ptx_version >= 35)
            return cta_dim_t{sm_35::nt, sm_35::vt};
        else if (ptx_version == 32)
            return cta_dim_t{sm_32::nt, sm_32::vt};
        else if (ptx_version >= 30)
            return cta_dim_t{sm_30::nt, sm_30::vt};
        else if (ptx_version >= 21)
            return cta_dim_t{sm_21::nt, sm_21::vt};
        else if (ptx_version >= 20)
            return cta_dim_t{sm_20::nt, sm_20::vt};
        else
            return cta_dim_t{-1, 0};
    }

    static cta_dim_t cta_dim(const Stream &stream) {
        return cta_dim(stream.device().info().ptx_version);
    }

    static int nv(const Stream &stream) {
        return cta_dim(stream.device().info().ptx_version).nv();
    }
};

template<typename launch_box, typename func_t, typename... args_t>
int occupancy(func_t f, const Stream &stream, args_t... args) {
    int num_blocks;
    int nt = launch_box::cta_dim(stream).nt;
    check_cuda_result(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &num_blocks,
        &launch_box_cta_k<launch_box, func_t, args_t...>,
        nt,
        (size_t)0));
    return stream.device().info().props.multiProcessorCount * num_blocks;
}

}// namespace vox