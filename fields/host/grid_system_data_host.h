//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "../grid_system_data.h"
#include "grid_data_host.h"

namespace vox::fields {
template<typename TYPE, uint32_t order, uint32_t dos>
class GridSystemData {
public:
    //! Constructs empty grid system.
    explicit GridSystemData(GridPtr<TYPE> grid);
    ~GridSystemData() = default;

    void sync_h2d();

    grid_system_data_t<TYPE, order, dos> view();

protected:
    const GridPtr<TYPE> _grid;
    std::array<GridDataPtr<TYPE, order>, dos> _scalarDataList;
};
}// namespace vox::fields