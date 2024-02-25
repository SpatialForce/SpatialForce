//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/cuda_tensor_view.h"

namespace vox::fields {
/// the data to describe a geometry.
struct geometry_t {
    /// Index of the geometry.
    CudaTensorView1<int32_t> ind{};
    /// Index of vertices.
    CudaTensorView1<uint32_t> vtx_index;
    CudaTensorView1<uint32_t> vtx;
    /// Index of boundary geometries.
    CudaTensorView1<uint32_t> bnd_index;
    CudaTensorView1<uint32_t> bnd;
    /// Boundary marker.
    CudaTensorView1<int32_t> bm{};

    [[nodiscard]] inline CUDA_CALLABLE_DEVICE int32_t index(uint32_t idx) const {
        return ind[idx];
    }

    /// Number of vertices.
    [[nodiscard]] inline CUDA_CALLABLE_DEVICE uint32_t n_vertex(uint32_t idx) const {
        if (idx == 0) {
            return vtx_index[0];
        } else {
            return vtx_index[idx] - vtx_index[idx - 1];
        }
    }

    /// The vertex index array.
    [[nodiscard]] inline CUDA_CALLABLE_DEVICE ConstCudaTensorView1<uint32_t> vertex(uint32_t idx) const {
        if (idx == 0) {
            return {vtx.data(), (size_t)vtx_index[0]};
        } else {
            return {vtx.data() + vtx_index[idx - 1], size_t(vtx_index[idx] - vtx_index[idx - 1])};
        }
    }

    /// An entry of the vertex index array.
    [[nodiscard]] inline CUDA_CALLABLE_DEVICE uint32_t vertex(uint32_t idx, uint32_t j) const {
        if (idx == 0) {
            return *(vtx.data() + j);
        } else {
            return *(vtx.data() + vtx_index[idx - 1] + j);
        }
    }

    /// Number of boundary geometries.
    [[nodiscard]] inline CUDA_CALLABLE_DEVICE uint32_t n_boundary(uint32_t idx) const {
        if (idx == 0) {
            return bnd_index[0];
        } else {
            return bnd_index[idx] - bnd_index[idx - 1];
        }
    }

    /// The boundary geometry index array.
    [[nodiscard]] inline CUDA_CALLABLE_DEVICE ConstCudaTensorView1<uint32_t> boundary(uint32_t idx) const {
        if (idx == 0) {
            return {bnd.data(), (size_t)bnd_index[0]};
        } else {
            return {bnd.data() + bnd_index[idx - 1], size_t(bnd_index[idx] - bnd_index[idx - 1])};
        }
    }

    /// An entry of the boundary geometry index array.
    [[nodiscard]] inline CUDA_CALLABLE_DEVICE uint32_t boundary(uint32_t idx, uint32_t j) const {
        if (idx == 0) {
            return *(bnd.data() + j);
        } else {
            return *(bnd.data() + bnd_index[idx - 1] + j);
        }
    }

    /// Access to the boundary marker.
    [[nodiscard]] inline CUDA_CALLABLE_DEVICE int32_t boundary_mark(uint32_t idx) const {
        return bm[idx];
    }
};

template<uint32_t SIZE>
struct static_geometry_t {
    static constexpr uint32_t size = SIZE;
    /// Index of the geometry.
    int32_t ind{};
    /// Index of vertices.
    CudaStdArray<uint32_t, size> vtx{};
    /// Index of boundary geometries.
    CudaStdArray<uint32_t, size> bnd{};
};

}// namespace vox::fields
