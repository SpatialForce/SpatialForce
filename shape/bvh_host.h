//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "bvh.h"
#include "runtime/cuda_buffer.h"
#include "runtime/cuda_tensor_view.h"

namespace vox {
class BvhHost {
public:
    BVH *view{};

    BvhHost(CudaTensorView1<Vector3F> lowers, CudaTensorView1<Vector3F> uppers);

    void refit();

private:
    CudaBuffer<BVHPackedNodeHalf> node_lowers;
    CudaBuffer<BVHPackedNodeHalf> node_uppers;
    CudaBuffer<int> node_parents;
    CudaBuffer<int> node_counts;
    CudaBuffer<int> root;
    CudaBuffer<BVH> bvh;
};
}// namespace vox