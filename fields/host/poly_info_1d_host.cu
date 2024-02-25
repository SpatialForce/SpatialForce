//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "poly_info_1d_host.h"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

namespace vox::fields {
namespace {
template<int order>
struct BuildBasisFuncFunctor {
    inline CUDA_CALLABLE BuildBasisFuncFunctor(const grid_t<Interval> &grid,
                                               CudaTensorView1<CudaStdArray<float, PolyInfo<Interval, order>::n_unknown>> poly_constants) {
        size = grid.size;
        output = poly_constants;
    }

    template<typename Index>
    inline CUDA_CALLABLE void operator()(Index ele_idx) {
        for (int m = 1; m <= order; ++m) {
            float J0 = pow(size[ele_idx] / 2.0, m + 1);
            J0 -= pow(-size[ele_idx] / 2.0, m + 1);
            J0 /= float(m + 1);

            J0 /= size[ele_idx];
            output[ele_idx][m - 1] = J0;
        }
    }

private:
    CudaTensorView1<float> size;
    CudaTensorView1<CudaStdArray<float, PolyInfo<Interval, order>::n_unknown>> output;
};
}// namespace

template<int order>
void PolyInfo<Interval, order>::build_basis_func() {
    thrust::for_each(thrust::counting_iterator<size_t>(0), thrust::counting_iterator<size_t>(0) + grid->n_geometry(1),
                     BuildBasisFuncFunctor<order>(grid->grid_view(), view().poly_constants));
}

template<int order>
void PolyInfo<Interval, order>::sync_h2d() {
    poly_constants.sync_h2d();
}

template<int order>
poly_info_t<Interval, order> PolyInfo<Interval, order>::view() {
    poly_info_t<Interval, order> handle;
    handle.poly_constants = poly_constants.view();
    return handle;
}

template class PolyInfo<Interval, 1>;
template class PolyInfo<Interval, 2>;
template class PolyInfo<Interval, 3>;
}// namespace vox::fields