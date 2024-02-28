//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "bounding_box.h"
#include "intersect.h"

namespace vox {
struct BVHPackedNodeHalf {
    float x;
    float y;
    float z;
    unsigned int i : 31;
    unsigned int b : 1;

    CUDA_CALLABLE inline BVHPackedNodeHalf(const Vector3F &bound, int child, bool leaf) {
        x = bound[0];
        y = bound[1];
        z = bound[2];
        i = (unsigned int)child;
        b = (unsigned int)(leaf ? 1 : 0);
    }
};

struct bvh_query_t;

struct BVH {
    BVHPackedNodeHalf *node_lowers{};
    BVHPackedNodeHalf *node_uppers{};

    // used for fast refits
    int *node_parents{};
    int *node_counts{};

    int max_depth{};
    int max_nodes{};
    int num_nodes{};

    // pointer (CPU or GPU) to a single integer index in node_lowers, node_uppers
    // representing the root of the tree, this is not always the first node
    // for bottom-up builders
    int *root{};

    // item bounds are not owned by the BVH but by the caller
    Vector3F *item_lowers{};
    Vector3F *item_uppers{};
    int num_items{};

    CUDA_CALLABLE inline int num_bounds() const {
        return num_items;
    }

    CUDA_CALLABLE inline bvh_query_t bvh_query(bool is_ray, const Vector3F &lower, const Vector3F &upper);

    CUDA_CALLABLE inline bvh_query_t bvh_query_aabb(const Vector3F &lower, const Vector3F &upper);

    CUDA_CALLABLE inline bvh_query_t bvh_query_ray(const Vector3F &start, const Vector3F &dir);
};

// stores state required to traverse the BVH nodes that
// overlap with a query AABB.
struct bvh_query_t {
    CUDA_CALLABLE bvh_query_t()
        : bvh(),
          stack(),
          count(0),
          is_ray(false),
          input_lower(),
          input_upper(),
          bounds_nr(0) {}

    // Required for adjoint computations.
    CUDA_CALLABLE inline bvh_query_t &operator+=(const bvh_query_t &other) {
        return *this;
    }

    BVH *bvh;

    // BVH traversal stack:
    int stack[32];
    int count;

    // inputs
    bool is_ray;
    Vector3F input_lower;// start for ray
    Vector3F input_upper;// dir for ray

    int bounds_nr;

    CUDA_CALLABLE inline bool bvh_query_next(int &index) {
        BoundingBox3F input_bounds(input_lower, input_upper);

        // Navigate through the bvh, find the first overlapping leaf node.
        while (count) {
            const int node_index = stack[--count];
            BVHPackedNodeHalf node_lower = bvh->node_lowers[node_index];
            BVHPackedNodeHalf node_upper = bvh->node_uppers[node_index];

            Vector3F lower_pos(node_lower.x, node_lower.y, node_lower.z);
            Vector3F upper_pos(node_upper.x, node_upper.y, node_upper.z);
            BoundingBox3F current_bounds(lower_pos, upper_pos);

            if (is_ray) {
                float t = 0.0f;
                if (!intersect_ray_aabb(input_lower, input_upper, current_bounds.lowerCorner, current_bounds.upperCorner, t))
                    // Skip this box, it doesn't overlap with our ray.
                    continue;
            } else {
                if (!input_bounds.overlaps(current_bounds))
                    // Skip this box, it doesn't overlap with our target box.
                    continue;
            }

            const int left_index = node_lower.i;
            const int right_index = node_upper.i;

            if (node_lower.b) {
                // found leaf
                bounds_nr = left_index;
                index = left_index;
                return true;
            } else {

                stack[count++] = left_index;
                stack[count++] = right_index;
            }
        }
        return false;
    }

    CUDA_CALLABLE inline int iter_next() const {
        return bounds_nr;
    }

    CUDA_CALLABLE inline bool iter_cmp() {
        bool finished = bvh_query_next(bounds_nr);
        return finished;
    }

    CUDA_CALLABLE inline bvh_query_t iter_reverse() {
        // can't reverse BVH queries, users should not rely on traversal ordering
        return *this;
    }
};

CUDA_CALLABLE inline bvh_query_t BVH::bvh_query(bool is_ray, const Vector3F &lower, const Vector3F &upper) {
    // This routine traverses the BVH tree until it finds
    // the first overlapping bound.

    // initialize empty
    bvh_query_t query;

    query.bounds_nr = -1;
    query.bvh = this;
    query.is_ray = is_ray;

    // optimization: make the latest
    query.stack[0] = *root;
    query.count = 1;
    query.input_lower = lower;
    query.input_upper = upper;

    BoundingBox3F input_bounds(query.input_lower, query.input_upper);

    // Navigate through the bvh, find the first overlapping leaf node.
    while (query.count) {
        const int node_index = query.stack[--query.count];

        BVHPackedNodeHalf node_lower = node_lowers[node_index];
        BVHPackedNodeHalf node_upper = node_uppers[node_index];

        Vector3F lower_pos(node_lower.x, node_lower.y, node_lower.z);
        Vector3F upper_pos(node_upper.x, node_upper.y, node_upper.z);
        BoundingBox3F current_bounds(lower_pos, upper_pos);

        if (query.is_ray) {
            float t = 0.0f;
            if (!intersect_ray_aabb(query.input_lower, query.input_upper, current_bounds.lowerCorner, current_bounds.upperCorner, t))
                // Skip this box, it doesn't overlap with our ray.
                continue;
        } else {
            if (!input_bounds.overlaps(current_bounds))
                // Skip this box, it doesn't overlap with our target box.
                continue;
        }

        const int left_index = node_lower.i;
        const int right_index = node_upper.i;

        // Make bounds from this AABB
        if (node_lower.b) {
            // found very first leaf index.
            // Back up one level and return
            query.stack[query.count++] = node_index;
            return query;
        } else {
            query.stack[query.count++] = left_index;
            query.stack[query.count++] = right_index;
        }
    }

    return query;
}

CUDA_CALLABLE inline bvh_query_t BVH::bvh_query_aabb(const Vector3F &lower, const Vector3F &upper) {
    return bvh_query(false, lower, upper);
}

CUDA_CALLABLE inline bvh_query_t BVH::bvh_query_ray(const Vector3F &start, const Vector3F &dir) {
    return bvh_query(true, start, dir);
}

//--------------------------------------------------------------------------------------------------------------------------
CUDA_CALLABLE inline int clz(int x) {
    int n;
    if (x == 0) return 32;
    for (n = 0; ((x & 0x80000000) == 0); n++, x <<= 1)
        ;
    return n;
}

CUDA_CALLABLE inline uint32_t part1by2(uint32_t n) {
    n = (n ^ (n << 16)) & 0xff0000ff;
    n = (n ^ (n << 8)) & 0x0300f00f;
    n = (n ^ (n << 4)) & 0x030c30c3;
    n = (n ^ (n << 2)) & 0x09249249;

    return n;
}

// Takes values in the range [0, 1] and assigns an index based Morton codes of length 3*lwp2(dim) bits
template<int dim>
CUDA_CALLABLE inline uint32_t morton3(float x, float y, float z) {
    uint32_t ux = clamp(int(x * dim), 0, dim - 1);
    uint32_t uy = clamp(int(y * dim), 0, dim - 1);
    uint32_t uz = clamp(int(z * dim), 0, dim - 1);

    return (part1by2(uz) << 2) | (part1by2(uy) << 1) | part1by2(ux);
}

}// namespace vox