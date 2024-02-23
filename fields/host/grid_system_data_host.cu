//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "grid_system_data_host.h"

namespace vox::fields {
template<typename TYPE, uint32_t order, uint32_t dos>
GridSystemData<TYPE, order, dos>::GridSystemData(GridPtr<TYPE> grid)
    : _grid{grid} {}

template<typename TYPE, uint32_t order, uint32_t dos>
void GridSystemData<TYPE, order, dos>::sync_h2d() {
    for (int i = 0; i < dos; i++) {
        _scalarDataList[i]->sync_h2d();
    }
}

template<typename TYPE, uint32_t order, uint32_t dos>
grid_system_data_t<TYPE, order, dos> GridSystemData<TYPE, order, dos>::view() {
    grid_system_data_t<TYPE, order, dos> handle;
    for (int i = 0; i < dos; i++) {
        handle.scalar_data_list[i] = _scalarDataList[i]->view();
    }
    return handle;
}

template<>
class GridSystemData<Interval, 1, 1>;
template<>
class GridSystemData<Triangle, 1, 1>;
template<>
class GridSystemData<Tetrahedron, 1, 1>;

}// namespace vox::fields