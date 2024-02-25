//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/matrix.h"
#include "fields/geometry.h"

namespace vox::fields {
/// The data structure of a mesh. The class \p Mesh administrate a set of points and
/// a set of geometries. The geometries are organized according its dimension and stored
/// in arrays. A lot of mechanism provided to retrieve information from the mesh.
template<uint32_t DIM, uint32_t DOW>
struct mesh_t {
    static constexpr uint32_t dim = DIM;
    static constexpr uint32_t dow = DOW;

    /// Point array of the mesh.
    CudaTensorView1<Vector<float, dow>> pnt;

    /// Geometries arrays of the mesh.
    /// The geometries in \p n dimension are in the \p n-th entry of the array,
    /// which is still an array.
    CudaStdArray<geometry_t, dim + 1> geo;

    /// Geometries array in certain dimension.
    [[nodiscard]] CUDA_CALLABLE_DEVICE const geometry_t &geometry(int d) const {
        return geo[d];
    }

    /// Geometries array in certain dimension.
    CUDA_CALLABLE_DEVICE geometry_t &geometry(int d) {
        return geo[d];
    }

    /// Boundary marker of certain geometry in certain dimension.
    [[nodiscard]] CUDA_CALLABLE_DEVICE auto boundary_mark(int d, int index) const {
        return geo[d].boundary_mark(index);
    }

    /// Boundary marker of certain geometry in certain dimension.
    CUDA_CALLABLE_DEVICE auto &boundary_mark(int d, int index) {
        return geo[d].boundary_mark(index);
    }
};

}// namespace vox::fields
