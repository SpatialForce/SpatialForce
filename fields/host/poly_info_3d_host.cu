//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "poly_info_3d_host.h"
#include "template_geometry.h"
#include "volume_integrator.h"
#include "runtime/cuda_util.h"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace vox::fields {
namespace {
template<int order>
struct BuildBasisFuncFunctor {
    using point_t = grid_t<Tetrahedron>::point_t;

    inline CUDA_CALLABLE
    BuildBasisFuncFunctor(const grid_t<Tetrahedron> &grid,
                          CudaTensorView1<CudaStdArray<float, PolyInfo<Tetrahedron, order>::n_unknown>> poly_constants)
        : grid(grid) {
        output = poly_constants;
    }

    struct IntegratorFunctor {
        CUDA_CALLABLE IntegratorFunctor(int j, int t, int k, int ele_idx, CudaTensorView1<point_t> bary_center)
            : j(j), t(t), k(k), ele_idx(ele_idx), bary_center(bary_center) {}

        using RETURN_TYPE = float;
        CUDA_CALLABLE float operator()(point_t pt) {
            Grid3D::point_t bc = bary_center(ele_idx);
            return pow(pt[0] - bc[0], j) * pow(pt[1] - bc[1], t) * pow(pt[2] - bc[2], k);
        }

        CudaTensorView1<point_t> bary_center;
        int j{};
        int t{};
        int k{};
        int ele_idx{};
    };

    template<typename Index>
    inline CUDA_CALLABLE void operator()(Index ele_idx) {
        int index = 0;
        float J0 = 0;
        for (int m = 1; m <= order; ++m) {
            for (int j = 0; j <= m; ++j) {
                for (int t = 0; t <= m - j; ++t) {
                    int k = m - j - t;
                    J0 = grid.volume_integrator(ele_idx, IntegratorFunctor(j, t, k, ele_idx, bary_center));
                    J0 /= area_of_ele(ele_idx);
                    output[ele_idx][index] = J0;
                    index++;
                }
            }
        }
    }

private:
    CudaTensorView1<float> area_of_ele;
    CudaTensorView1<point_t> bary_center;
    CudaTensorView1<CudaStdArray<float, PolyInfo<Tetrahedron, order>::n_unknown>> output;
    grid_t<Tetrahedron> grid;
};
}// namespace

template<int order>
void PolyInfo<Tetrahedron, order>::build_basis_func() {
    thrust::for_each(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(0) + grid->n_geometry(1),
                     BuildBasisFuncFunctor<order>(grid->grid_view(), view().poly_constants));
}

template<int order>
void PolyInfo<Tetrahedron, order>::sync_h2d() {
    poly_constants.sync_h2d();
}

template<int order>
poly_info_t<Tetrahedron, order> PolyInfo<Tetrahedron, order>::view() {
    poly_info_t<Tetrahedron, order> handle;
    handle.poly_constants = poly_constants.view();
    return handle;
}

template class PolyInfo<Tetrahedron, 1>;
template class PolyInfo<Tetrahedron, 2>;
template class PolyInfo<Tetrahedron, 3>;
}// namespace vox::fields