//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>
#include "runtime/cuda_tensor.h"
#include "../geometry.h"

namespace vox::fields {
class Geometry {
public:
    geometry_t view();

    [[nodiscard]] uint32_t n_index() const;

    [[nodiscard]] int32_t index(uint32_t) const;
    /// Number of vertices.
    [[nodiscard]] uint32_t n_vertex(uint32_t) const;
    /// An entry of the vertex index array.
    [[nodiscard]] uint32_t vertex(uint32_t, uint32_t) const;

    /// Number of boundary geometries.
    [[nodiscard]] uint32_t n_boundary(uint32_t) const;
    /// An entry of the boundary geometry index array.
    [[nodiscard]] uint32_t boundary(uint32_t, uint32_t) const;
    /// Access to the boundary marker.
    [[nodiscard]] int32_t boundary_mark(uint32_t) const;

    void sync_h2d();

public:
    /// Index of the geometry.
    CudaTensor1<int32_t> ind;
    /// Index of vertices.
    CudaTensor1<uint32_t> vtx_index;
    CudaTensor1<uint32_t> vtx;
    /// Index of boundary geometries.
    CudaTensor1<uint32_t> bnd_index;
    CudaTensor1<uint32_t> bnd;
    /// Boundary marker.
    CudaTensor1<int32_t> bm;
};
}// namespace vox::fields