//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/matrix.h"
#include "poly_info.h"
#include "grid.h"

namespace vox::fields {

template<uint32_t ORDER>
struct poly_info_t<Triangle, ORDER> {
    static constexpr uint32_t dim = Triangle::dim;
    static constexpr uint32_t order = ORDER;
    static constexpr uint32_t n_unknown = (order + 2) * (order + 1) / 2 - 1;
    CudaTensorView1<CudaStdArray<float, n_unknown>> poly_constants;
    using grid_t = grid_t<Triangle>;
    using point_t = grid_t::point_t;
    using Mat = Matrix<float, n_unknown, n_unknown>;
    using Vec = Matrix<float, n_unknown, 1>;

    struct AverageBasisFuncFunctor {
        CUDA_CALLABLE AverageBasisFuncFunctor(grid_t grid, poly_info_t<Triangle, order> poly)
            : grid(std::move(grid)), poly_constants{poly.poly_constants} {
        }

        struct IntegratorFunctor {
            CUDA_CALLABLE IntegratorFunctor(int j, int k, point_t bc)
                : j(j), k(k), bc(bc) {}

            using RETURN_TYPE = float;
            CUDA_CALLABLE_DEVICE float operator()(point_t pt) {
                return pow(pt[0] - bc[0], j) * pow(pt[1] - bc[1], k);
            }

            point_t bc;
            int j{};
            int k{};
        };

        /// Element average of basis function
        /// @param basisIdx the loc of basis function
        /// @param quadIdx the loc of quadrature area
        /// @param result result
        CUDA_CALLABLE_DEVICE void operator()(uint32_t basisIdx, int32_t quadIdx, CudaStdArray<float, n_unknown> &result) {
            point_t bc = grid.bary_center(basisIdx);
            int index = 0;
            float J0 = 0;
            for (int m = 1; m <= order; ++m) {
                for (int i = 0; i <= m; ++i) {
                    int k = m - i;
                    J0 = grid.volume_integrator(quadIdx, IntegratorFunctor(i, k, bc));
                    J0 /= grid.volume(quadIdx);
                    J0 -= poly_constants(basisIdx)[index];
                    J0 *= pow(grid.bry_size(basisIdx), 1.f - float(m));
                    result[index] = J0;
                    index++;
                }
            }
        }

        CUDA_CALLABLE_DEVICE void operator()(uint32_t basisIdx, CudaTensorView1<int32_t> patch,
                                      CudaTensorView1<CudaStdArray<float, n_unknown>> result) {
            CudaStdArray<float, n_unknown> s;
            for (uint32_t j = 0; j < patch.width(); ++j) {
                operator()(basisIdx, patch[j], s);
                result[j] = s;
            }
        }

    private:
        grid_t grid;
        CudaTensorView1<CudaStdArray<float, n_unknown>> poly_constants;
    };

    struct UpdateLSMatrixFunctor {
        CUDA_CALLABLE UpdateLSMatrixFunctor(grid_t grid, poly_info_t<Triangle, order> poly)
            : averageBasisFunc(grid, poly) {
        }

        CUDA_CALLABLE_DEVICE void operator()(uint32_t basisIdx, const CudaTensorView1<int32_t> &patch,
                                      CudaTensorView1<CudaStdArray<float, n_unknown>> poly_avgs, Mat *G) {
            averageBasisFunc(basisIdx, patch, poly_avgs);

            G[0].fill(0.f);
            for (uint32_t j = 0; j < patch.width(); ++j) {
                CudaStdArray<float, n_unknown> poly_avg = poly_avgs[j];
                for (int t1 = 0; t1 < n_unknown; ++t1) {
                    for (int t2 = 0; t2 < n_unknown; ++t2) {
                        G[0](t1, t2) += poly_avg[t1] * poly_avg[t2];
                    }
                }
            }
        }

    private:
        AverageBasisFuncFunctor averageBasisFunc;
    };

    struct FuncValueFunctor {
        CUDA_CALLABLE_DEVICE float operator()(size_t idx, const point_t &coord, const float &avg, const Vec &para) {
            CudaStdArray<float, n_unknown> aa;
            basis_function_value(idx, coord, aa);

            float temp = 0.0;
            float result = avg;
            for (int t = 0; t < n_unknown; ++t) {
                temp = para[t] * aa[t];
                result += temp;
            }
            return result;
        }

        CUDA_CALLABLE_DEVICE void basis_function_value(size_t idx, const point_t &coord,
                                                CudaStdArray<float, n_unknown> &result) {
            point_t cr = coord;
            cr -= bary_center(idx);
            int index = 0;
            float J0 = 0;
            for (int m = 1; m <= order; ++m)
                for (int i = 0; i <= m; ++i) {
                    int j = m - i;
                    J0 = pow(cr[0], float(i)) * pow(cr[1], float(j));
                    J0 -= poly_constants(idx)[index];
                    J0 *= pow(bary_size(idx), 1.f - float(m));
                    result[index] = J0;
                    index++;
                }
        }

        CudaTensorView1<point_t> bary_center;
        CudaTensorView1<float> bary_size;

        CudaTensorView1<CudaStdArray<float, n_unknown>> poly_constants;
    };

    struct FuncGradientFunctor {
        CUDA_CALLABLE_DEVICE Vector<float, dim> operator()(size_t idx, const point_t &coord,
                                                    const Vec &para) {
            CudaStdArray<CudaStdArray<float, n_unknown>, dim> aa;
            basis_function_gradient(idx, coord, aa);

            float temp = 0.0;
            Vector<float, dim> result;
            for (uint32_t i = 0; i < dim; ++i) {
                for (int t = 0; t < n_unknown; ++t) {
                    temp = para[t] * aa[i][t];
                    result[i] += temp;
                }
            }

            return result;
        }

        CUDA_CALLABLE_DEVICE void basis_function_gradient(size_t idx, const point_t &coord,
                                                   CudaStdArray<CudaStdArray<float, n_unknown>, dim> &result) {
            point_t cr = coord;
            cr -= bary_center(idx);

            // coordinate x
            int index = 0;
            float J0 = 0.0;
            for (int m = 1; m <= order; ++m) {
                for (int i = 0; i <= m; ++i) {
                    int j = m - i;
                    if (i == 0) {
                        J0 = 0.0;
                    } else {
                        J0 = float(i) * pow(cr[0], float(i) - 1) * pow(cr[1], float(j));
                    }
                    J0 *= pow(bary_size(idx), 1.f - float(m));
                    result[0][index] = J0;
                    index++;
                }
            }

            // coordinate y
            index = 0;
            J0 = 0.0;
            for (int m = 1; m <= order; ++m) {
                for (int i = 0; i <= m; ++i) {
                    int j = m - i;
                    if (j == 0) {
                        J0 = 0.0;
                    } else {
                        J0 = pow(cr[0], float(i)) * float(j) * pow(cr[1], float(j) - 1);
                    }
                    J0 *= pow(bary_size(idx), 1.f - float(m));
                    result[1][index] = J0;
                    index++;
                }
            }
        }

        CudaTensorView1<point_t> bary_center;
        CudaTensorView1<float> bary_size;
    };
};

}// namespace vox::fields