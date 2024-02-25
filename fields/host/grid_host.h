//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "../grid.h"
#include "../mesh.h"
#include <string>
#include <memory>
#include "mesh_host.h"

namespace vox::fields {
template<typename TYPE>
class Grid {
public:
    static constexpr uint32_t dim = TYPE::dim;
    using point_t = Vector<float, dim>;

    mesh_t<dim, dim> mesh_view() {
        return mesh.view();
    }
    grid_t<TYPE> grid_view() {
        grid_t<TYPE> handle;
        handle.bary_center = bary_center.view();
        handle.volume = volume.view();
        handle.size = size.view();
        handle.neighbour = neighbour.view();
        handle.period_bry = period_bry.view();
        handle.boundary_center = boundary_center.view();
        handle.bry_size = bry_size.view();
        handle.boundary_mark = boundary_mark.view();
        return handle;
    }

    void sync_h2d() {
        bary_center.sync_h2d();
        volume.sync_h2d();
        size.sync_h2d();
        neighbour.sync_h2d();
        period_bry.sync_h2d();
        boundary_center.sync_h2d();
        bry_size.sync_h2d();
        boundary_mark.sync_h2d();

        mesh.sync_h2d();
    }

    /// Number of geometries in certain dimension.
    [[nodiscard]] uint32_t n_geometry(int n) const {
        return mesh.n_geometry(n);
    }

private:
    Mesh<dim, dim> mesh;
    HostDeviceVector<point_t> bary_center;
    HostDeviceVector<float> volume;
    HostDeviceVector<float> size;

    HostDeviceVector<CudaStdArray<int32_t, 2>> neighbour;
    HostDeviceVector<CudaStdArray<int32_t, 2>> period_bry;
    HostDeviceVector<point_t> boundary_center;
    HostDeviceVector<float> bry_size;
    HostDeviceVector<int32_t> boundary_mark;
};

using Grid1D = Grid<Interval>;
using Grid2D = Grid<Triangle>;
using Grid3D = Grid<Tetrahedron>;

template<typename TYPE>
using GridPtr = std::shared_ptr<Grid<TYPE>>;

using GridPtr1D = std::shared_ptr<Grid1D>;
using GridPtr2D = std::shared_ptr<Grid2D>;
using GridPtr3D = std::shared_ptr<Grid3D>;
}// namespace vox::fields