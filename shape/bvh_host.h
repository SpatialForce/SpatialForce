//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "bvh.h"
#include "runtime/cuda_buffer.h"

namespace vox {
class BvhHost {
public:
    BVH *view;

    BvhHost(Vector3F *lowers, Vector3F *uppers, int num_items);

    void refit();

private:
    CudaBuffer<BVHPackedNodeHalf> node_lowers;
    CudaBuffer<BVHPackedNodeHalf> node_uppers;
    CudaBuffer<int> node_parents;
    CudaBuffer<int> node_counts;
    CudaBuffer<int> root;
};
}// namespace vox