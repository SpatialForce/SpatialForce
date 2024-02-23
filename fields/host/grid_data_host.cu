//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "grid_data_host.h"

namespace vox::fields {
template<typename TYPE>
void GridDataSimple<TYPE>::sync_h2d() {
    data.sync_h2d();
}

template<typename TYPE>
grid_data_base_t<TYPE> GridDataSimple<TYPE>::view() {
    grid_data_base_t<TYPE> handle;
    handle.data = data.view();
    return handle;
}

template class GridDataSimple<Interval>;
template class GridDataSimple<Triangle>;
template class GridDataSimple<Tetrahedron>;

template<typename TYPE, uint32_t order>
GridData<TYPE, order>::GridData(uint32_t idx, GridPtr<TYPE> grid, ReconAuxiliaryPtr<TYPE, order> aux)
    : GridDataBase{idx}, grid{grid}, recon_auxiliary{aux} {
}

template<typename TYPE, uint32_t order>
void GridData<TYPE, order>::sync_h2d() {
    data.sync_h2d();
    slope.sync_h2d();
}

template<typename TYPE, uint32_t order>
grid_data_t<TYPE, order> GridData<TYPE, order>::view() {
    grid_data_t<TYPE, order> handle;
    handle.data = data.view();
    handle.slope = slope.view();
    return handle;
}

template class GridData<Interval, 1>;
template class GridData<Triangle, 1>;
template class GridData<Tetrahedron, 1>;
}// namespace vox::fields