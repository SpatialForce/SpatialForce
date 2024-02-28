//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "bvh_host.h"

namespace vox {
namespace {
// Create a linear BVH as described in Fast and Simple Agglomerative LBVH construction
// this is a bottom-up clustering method that outputs one node per-leaf
//
class LinearBVHBuilderGPU {
public:
    LinearBVHBuilderGPU() {}
    ~LinearBVHBuilderGPU() {}

    // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for each triangle
    void build(BVH &bvh, const Vector3F *item_lowers, const Vector3F *item_uppers, int num_items, BoundingBox3F *total_bounds) {}

private:

    // temporary data used during building
    int *indices;
    int *keys;
    int *deltas;
    int *range_lefts;
    int *range_rights;
    int *num_children;

    // bounds data when total item bounds built on GPU
    Vector3F *total_lower;
    Vector3F *total_upper;
    Vector3F *total_inv_edges;
};
}// namespace

BvhHost::BvhHost(Vector3F *lowers, Vector3F *uppers, int num_items) {}

void BvhHost::refit() {}
}// namespace vox