//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "poly_info_host.h"
#include "grid_host.h"
#include "../poly_info_2d.h"

namespace vox::fields {
template<int order>
class PolyInfo<Triangle, order> {
public:
    static constexpr int n_unknown = (order + 2) * (order + 1) / 2 - 1;

    poly_info_t<Triangle, order> view();

    explicit PolyInfo(GridPtr2D grid) : grid{std::move(grid)} {
        build_basis_func();
        sync_h2d();
    }

    ~PolyInfo() = default;

private:
    void build_basis_func();
    void sync_h2d();

    GridPtr2D grid;
    HostDeviceVector<fixed_array_t<float, n_unknown>> poly_constants;
};

}// namespace vox::fields