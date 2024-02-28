//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "bvh_host.h"
#include "radix_sort.h"
#include "runtime/transform.h"
#include <cub/cub.cuh>

namespace vox {
namespace {
CUDA_CALLABLE_DEVICE void compute_morton_codes(const Vector3F *__restrict__ item_lowers, const Vector3F *__restrict__ item_uppers, int n,
                                               const Vector3F *grid_lower, const Vector3F *grid_inv_edges,
                                               int *__restrict__ indices, int *__restrict__ keys) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        Vector3F lower = item_lowers[index];
        Vector3F upper = item_uppers[index];

        Vector3F center = 0.5f * (lower + upper);

        Vector3F local = elemMul((center - grid_lower[0]), grid_inv_edges[0]);

        // 10-bit Morton codes stored in lower 30bits (1024^3 effective resolution)
        int key = morton3<1024>(local[0], local[1], local[2]);

        indices[index] = index;
        keys[index] = key;
    }
}

// calculate the index of the first differing bit between two adjacent Morton keys
CUDA_CALLABLE_DEVICE void compute_key_deltas(const int *__restrict__ keys, int *__restrict__ deltas, int n) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        int a = keys[index];
        int b = keys[index + 1];

        int x = a ^ b;

        deltas[index] = x;// __clz(x);
    }
}

CUDA_CALLABLE_DEVICE void build_leaves(const Vector3F *__restrict__ item_lowers, const Vector3F *__restrict__ item_uppers, int n,
                                       const int *__restrict__ indices, int *__restrict__ range_lefts, int *__restrict__ range_rights,
                                       BVHPackedNodeHalf *__restrict__ lowers, BVHPackedNodeHalf *__restrict__ uppers) {
    const int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        const int item = indices[index];

        Vector3F lower = item_lowers[item];
        Vector3F upper = item_uppers[item];

        // write leaf nodes
        lowers[index] = BVHPackedNodeHalf(lower, item, true);
        uppers[index] = BVHPackedNodeHalf(upper, item, false);

        // write leaf key ranges
        range_lefts[index] = index;
        range_rights[index] = index;
    }
}

// this bottom-up process assigns left and right children and combines bounds to form internal nodes
// there is one thread launched per-leaf node, each thread calculates it's parent node and assigns
// itself to either the left or right parent slot, the last child to complete the parent and moves
// up the hierarchy
CUDA_CALLABLE_DEVICE void build_hierarchy(int n, int *root, const int *__restrict__ deltas, int *__restrict__ num_children,
                                          volatile int *__restrict__ range_lefts, volatile int *__restrict__ range_rights,
                                          volatile int *__restrict__ parents,
                                          BVHPackedNodeHalf *__restrict__ lowers,
                                          BVHPackedNodeHalf *__restrict__ uppers) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        const int internal_offset = n;

        for (;;) {
            int left = range_lefts[index];
            int right = range_rights[index];

            // check if we are the root node, if so then store out our index and terminate
            if (left == 0 && right == n - 1) {
                *root = index;
                parents[index] = -1;

                break;
            }

            int childCount = 0;

            int parent;

            if (left == 0 || (right != n - 1 && deltas[right] < deltas[left - 1])) {
                parent = right + internal_offset;

                // set parent left child
                parents[index] = parent;
                lowers[parent].i = index;
                range_lefts[parent] = left;

                // ensure above writes are visible to all threads
                __threadfence();

                childCount = atomicAdd(&num_children[parent], 1);
            } else {
                parent = left + internal_offset - 1;

                // set parent right child
                parents[index] = parent;
                uppers[parent].i = index;
                range_rights[parent] = right;

                // ensure above writes are visible to all threads
                __threadfence();

                childCount = atomicAdd(&num_children[parent], 1);
            }

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the the next parent in the hierarchy
            if (childCount == 1) {
                const int left_child = lowers[parent].i;
                const int right_child = uppers[parent].i;

                Vector3F left_lower = Vector3F(lowers[left_child].x,
                                               lowers[left_child].y,
                                               lowers[left_child].z);

                Vector3F left_upper = Vector3F(uppers[left_child].x,
                                               uppers[left_child].y,
                                               uppers[left_child].z);

                Vector3F right_lower = Vector3F(lowers[right_child].x,
                                                lowers[right_child].y,
                                                lowers[right_child].z);

                Vector3F right_upper = Vector3F(uppers[right_child].x,
                                                uppers[right_child].y,
                                                uppers[right_child].z);

                // bounds_union of child bounds
                Vector3F lower = min(left_lower, right_lower);
                Vector3F upper = max(left_upper, right_upper);

                // write new BVH nodes
                lowers[parent] = BVHPackedNodeHalf(lower, left_child, false);
                uppers[parent] = BVHPackedNodeHalf(upper, right_child, false);

                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
}

CUDA_CALLABLE inline Vector3F Vec3Max(const Vector3F &a, const Vector3F &b) { return max(a, b); }
CUDA_CALLABLE inline Vector3F Vec3Min(const Vector3F &a, const Vector3F &b) { return min(a, b); }

CUDA_CALLABLE_DEVICE void compute_total_bounds(const Vector3F *item_lowers, const Vector3F *item_uppers,
                                               Vector3F *total_lower, Vector3F *total_upper, int num_items) {
    typedef cub::BlockReduce<Vector3F, 256> BlockReduce;

    __shared__ typename BlockReduce::TempStorage temp_storage;

    const int blockStart = blockDim.x * blockIdx.x;
    const int numValid = ::min(num_items - blockStart, blockDim.x);

    const int tid = blockStart + threadIdx.x;

    if (tid < num_items) {
        Vector3F lower = item_lowers[tid];
        Vector3F upper = item_uppers[tid];

        Vector3F block_upper = BlockReduce(temp_storage).Reduce(upper, Vec3Max, numValid);

        // sync threads because second reduce uses same temp storage as first
        __syncthreads();

        Vector3F block_lower = BlockReduce(temp_storage).Reduce(lower, Vec3Min, numValid);

        if (threadIdx.x == 0) {
            // write out block results, expanded by the radius
            atomic_max(*total_upper, block_upper);
            atomic_min(*total_lower, block_lower);
        }
    }
}

// compute inverse edge length, this is just done on the GPU to avoid a CPU->GPU sync point
CUDA_CALLABLE_DEVICE void compute_total_inv_edges(const Vector3F *total_lower, const Vector3F *total_upper, Vector3F *total_inv_edges) {
    Vector3F edges = (total_upper[0] - total_lower[0]);
    edges += 0.0001f;

    total_inv_edges[0] = Vector3F(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);
}

// Create a linear BVH as described in Fast and Simple Agglomerative LBVH construction
// this is a bottom-up clustering method that outputs one node per-leaf
//
class LinearBVHBuilderGPU {
public:
    LinearBVHBuilderGPU() {
        total_lower.resize(1);
        total_upper.resize(1);
        total_inv_edges.resize(1);
    }

    // takes a bvh (host ref), and pointers to the GPU lower and upper bounds for each triangle
    void build(BVH &bvh, CudaTensorView1<Vector3F> lowers, CudaTensorView1<Vector3F> uppers, BoundingBox3F *total_bounds) {
        int num_items = lowers.width();
        indices.resize(num_items * 2);// *2 for radix sort
        keys.resize(num_items * 2);   // *2 for radix sort
        deltas.resize(num_items);     // highest differenting bit between keys for item i and i+1
        range_lefts.resize(bvh.max_nodes);
        range_rights.resize(bvh.max_nodes);
        num_children.resize(bvh.max_nodes);

        // if total bounds supplied by the host then we just
        // compute our edge length and upload it to the GPU directly
        if (total_bounds) {
            // calculate Morton codes
            Vector3F edges = total_bounds->size();
            edges += 0.0001f;

            Vector3F inv_edges = Vector3F(1.0f / edges[0], 1.0f / edges[1], 1.0f / edges[2]);

            // memcpy_h2d(WP_CURRENT_CONTEXT, total_lower, &total_bounds->lower[0], sizeof(vec3));
            // memcpy_h2d(WP_CURRENT_CONTEXT, total_upper, &total_bounds->upper[0], sizeof(vec3));
            // memcpy_h2d(WP_CURRENT_CONTEXT, total_inv_edges, &inv_edges[0], sizeof(vec3));
        } else {
            total_lower.fill(Vector3F(FLT_MAX, FLT_MAX, FLT_MAX));
            total_upper.fill(Vector3F(-FLT_MAX, -FLT_MAX, -FLT_MAX));

            // compute the total bounds on the GPU
            auto compute_total_bounds_callable = [item_lowers = lowers.data(), item_uppers = uppers.data(),
                                                  total_lower = total_lower.data(), total_upper = total_upper.data(),
                                                  num_items = (int)lowers.width()] CUDA_CALLABLE_DEVICE(int index) {
                compute_total_bounds(item_lowers, item_uppers, total_lower, total_upper, num_items);
            };
            transform(compute_total_bounds_callable, num_items, device().stream);

            // compute the total edge length
            auto compute_total_inv_edges_callable = [total_lower = total_lower.data(), total_upper = total_upper.data(),
                                                     total_inv_edges = total_inv_edges.data()] CUDA_CALLABLE_DEVICE(int index) {
                compute_total_inv_edges(total_lower, total_upper, total_inv_edges);
            };
            transform(compute_total_inv_edges_callable, 1, device().stream);
        }

        // assign 30-bit Morton code based on the centroid of each triangle and bounds for each leaf
        auto compute_morton_codes_callable = [item_lowers = lowers.data(), item_uppers = uppers.data(), num_items,
                                              total_lower = total_lower.data(),
                                              total_inv_edges = total_inv_edges.data(),
                                              indices = indices.data(),
                                              keys = keys.data()] CUDA_CALLABLE_DEVICE(int index) {
            compute_morton_codes(item_lowers, item_uppers, num_items, total_lower, total_inv_edges, indices, keys);
        };
        transform(compute_morton_codes_callable, num_items, device().stream);

        // sort items based on Morton key (note the 32-bit sort key corresponds to the template parameter to morton3, i.e. 3x9 bit keys combined)
        sort.execute(keys.data(), indices.data(), num_items);

        // calculate deltas between adjacent keys
        auto compute_key_deltas_callable = [num_items,
                                            keys = keys.data(),
                                            deltas = deltas.data()] CUDA_CALLABLE_DEVICE(int index) {
            compute_key_deltas(keys, deltas, num_items - 1);
        };
        transform(compute_key_deltas_callable, num_items, device().stream);

        // initialize leaf nodes
        auto build_leaves_callable = [item_lowers = lowers.data(), item_uppers = uppers.data(), num_items,
                                      indices = indices.data(),
                                      range_lefts = range_lefts.data(),
                                      range_rights = range_rights.data(),
                                      node_lowers = bvh.node_lowers,
                                      node_uppers = bvh.node_uppers] CUDA_CALLABLE_DEVICE(int index) {
            build_leaves(item_lowers, item_uppers, num_items, indices, range_lefts, range_rights, node_lowers, node_uppers);
        };
        transform(build_leaves_callable, num_items, device().stream);

        // reset children count, this is our atomic counter so we know when an internal node is complete, only used during building
        num_children.fill(0);

        // build the tree and internal node bounds
        auto build_hierarchy_callable = [root = bvh.root, num_items,
                                         num_children = num_children.data(),
                                         deltas = deltas.data(),
                                         range_lefts = range_lefts.data(),
                                         range_rights = range_rights.data(),
                                         node_parents = bvh.node_parents,
                                         node_lowers = bvh.node_lowers,
                                         node_uppers = bvh.node_uppers] CUDA_CALLABLE_DEVICE(int index) {
            build_hierarchy(num_items, root, deltas, num_children, range_lefts, range_rights, node_parents, node_lowers, node_uppers);
        };
        transform(build_hierarchy_callable, num_items, device().stream);
    }

private:
    RadixSort sort;

    // temporary data used during building
    CudaBuffer<int> indices;
    CudaBuffer<int> keys;
    CudaBuffer<int> deltas;
    CudaBuffer<int> range_lefts;
    CudaBuffer<int> range_rights;
    CudaBuffer<int> num_children;

    // bounds data when total item bounds built on GPU
    CudaBuffer<Vector3F> total_lower;
    CudaBuffer<Vector3F> total_upper;
    CudaBuffer<Vector3F> total_inv_edges;
};

CUDA_CALLABLE_DEVICE void bvh_refit_kernel(int n, const int *__restrict__ parents, int *__restrict__ child_count,
                                           BVHPackedNodeHalf *__restrict__ node_lowers, BVHPackedNodeHalf *__restrict__ node_uppers,
                                           const Vector3F *item_lowers, const Vector3F *item_uppers) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;

    if (index < n) {
        bool leaf = node_lowers[index].b;

        if (leaf) {
            // update the leaf node
            const int leaf_index = node_lowers[index].i;

            Vector3F lower = item_lowers[leaf_index];
            Vector3F upper = item_uppers[leaf_index];

            node_lowers[index] = BVHPackedNodeHalf(lower, leaf_index, true);
            node_uppers[index] = BVHPackedNodeHalf(upper, 0, false);
        } else {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;) {
            int parent = parents[index];

            // reached root
            if (parent == -1)
                return;

            // ensure all writes are visible
            __threadfence();

            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the the next parent in the hierarchy
            if (finished == 1) {
                const int left_child = node_lowers[parent].i;
                const int right_child = node_uppers[parent].i;

                Vector3F left_lower = Vector3F(node_lowers[left_child].x,
                                               node_lowers[left_child].y,
                                               node_lowers[left_child].z);

                Vector3F left_upper = Vector3F(node_uppers[left_child].x,
                                               node_uppers[left_child].y,
                                               node_uppers[left_child].z);

                Vector3F right_lower = Vector3F(node_lowers[right_child].x,
                                                node_lowers[right_child].y,
                                                node_lowers[right_child].z);

                Vector3F right_upper = Vector3F(node_uppers[right_child].x,
                                                node_uppers[right_child].y,
                                                node_uppers[right_child].z);

                // union of child bounds
                Vector3F lower = min(left_lower, right_lower);
                Vector3F upper = max(left_upper, right_upper);

                // write new BVH nodes
                node_lowers[parent] = BVHPackedNodeHalf(lower, left_child, false);
                node_uppers[parent] = BVHPackedNodeHalf(upper, right_child, false);

                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
}

}// namespace

BvhHost::BvhHost(CudaTensorView1<Vector3F> lowers, CudaTensorView1<Vector3F> uppers) {
    BVH bvh_host;
    bvh_host.item_lowers = lowers.data();
    bvh_host.item_uppers = uppers.data();

    LinearBVHBuilderGPU builder;
    builder.build(bvh_host, lowers, uppers, nullptr);

    bvh = CudaBuffer<BVH>(1, bvh_host);
    view = bvh.data();
}

void BvhHost::refit() {
    auto k = [this] CUDA_CALLABLE_DEVICE(int index) {
        bvh_refit_kernel(view->num_items, view->node_parents, view->node_counts,
                         view->node_lowers, view->node_uppers,
                         view->item_lowers, view->item_uppers);
    };
    transform(k, view->num_items, device().stream);
}
}// namespace vox