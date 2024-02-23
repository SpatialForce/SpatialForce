//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "geometry_host.h"

namespace vox::fields {
uint32_t Geometry::n_index() const {
    return ind.host_buffer.size();
}

int32_t Geometry::index(uint32_t idx) const {
    return ind.host_buffer[idx];
}
uint32_t Geometry::n_vertex(uint32_t idx) const {
    if (idx == 0) {
        return vtx_index.host_buffer[0];
    } else {
        return vtx_index.host_buffer[idx] - vtx_index.host_buffer[idx - 1];
    }
}

uint32_t Geometry::vertex(uint32_t idx, uint32_t j) const {
    if (idx == 0) {
        return *(vtx.host_buffer.data() + j);
    } else {
        return *(vtx.host_buffer.data() + vtx_index.host_buffer[idx - 1] + j);
    }
}

uint32_t Geometry::n_boundary(uint32_t idx) const {
    if (idx == 0) {
        return bnd_index.host_buffer[0];
    } else {
        return bnd_index.host_buffer[idx] - bnd_index.host_buffer[idx - 1];
    }
}
uint32_t Geometry::boundary(uint32_t idx, uint32_t j) const {
    if (idx == 0) {
        return *(bnd.host_buffer.data() + j);
    } else {
        return *(bnd.host_buffer.data() + bnd_index.host_buffer[idx - 1] + j);
    }
}
int32_t Geometry::boundary_mark(uint32_t idx) const {
    return bm.host_buffer[idx];
}

void Geometry::sync_h2d() {
    ind.sync_h2d();
    vtx_index.sync_h2d();
    vtx.sync_h2d();
    bnd_index.sync_h2d();
    bnd.sync_h2d();
    bm.sync_h2d();
}

geometry_t Geometry::view() {
    geometry_t handle;
    handle.ind = ind.view();
    handle.vtx_index = vtx_index.view();
    handle.vtx = vtx.view();
    handle.bnd_index = bnd_index.view();
    handle.bnd = bnd.view();
    handle.bm = bm.view();
    return handle;
}

}// namespace vox::fields