//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "../mesh.h"
#include "geometry_host.h"
#include <vector>

namespace vox::fields {
template<uint32_t DIM, uint32_t DOW>
class IOMesh;

template<uint32_t DIM, uint32_t DOW>
class Mesh {
public:
    static constexpr uint32_t dim = DIM;
    static constexpr uint32_t dow = DOW;
    using point_t = Vector<float, dow>;

    mesh_t<dim, dow> view() {
        mesh_t<dim, dow> handle;
        handle.pnt = pnt.view();
        for (int i = 0; i <= dim; i++) {
            handle.geo[i] = geo[i].view();
        }
        return handle;
    }

    /// Number of points in the mesh.
    [[nodiscard]] uint32_t n_point() const {
        return pnt.host_buffer.size();
    }

    /// Number of geometries in certain dimension.
    [[nodiscard]] uint32_t n_geometry(int n) const {
        return geo[n].n_index();
    }

    /// Point array.
    const std::vector<point_t> &point() const {
        return pnt.host_buffer;
    }

    /// Point array.
    std::vector<point_t> &point() {
        return pnt.host_buffer;
    }

    /// A certain point.
    const point_t &point(int i) const {
        return pnt.host_buffer[i];
    }

    /// A certain point.
    point_t &point(int i) {
        return pnt.host_buffer[i];
    }

    /// Geometries array in certain dimension.
    [[nodiscard]] const Geometry &geometry(int n) const {
        return geo[n];
    }

    /// Geometries array in certain dimension.
    Geometry &geometry(int n) {
        return geo[n];
    }

    /// Boundary marker of certain geometry in certain dimension.
    [[nodiscard]] int32_t boundary_mark(int n, int j) const {
        return geo[n].boundary_mark(j);
    }

    void sync_h2d() {
        pnt.sync_h2d();
        for (int i = 0; i <= dim; i++) {
            geo[i].sync_h2d();
        }
    }

private:
    friend class IOMesh<dim, dow>;

    /// Point array of the mesh.
    HostDeviceVector<point_t> pnt;
    /// Geometries arrays of the mesh.
    /// The geometries in \p n dimension are in the \p n-th entry of the array,
    /// which is still an array.
    Geometry geo[dim + 1];
};

}// namespace vox::fields