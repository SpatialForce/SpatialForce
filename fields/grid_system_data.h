//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/matrix.h"
#include "grid_data.h"

namespace vox::fields {
template<typename TYPE, uint32_t order, uint32_t dos>
struct grid_system_data_t {
    static constexpr uint32_t dim = TYPE::dim;
    using point_t = Vector<float, dim>;
    using system_array_t = Vector<float, dos>;

    CudaStdArray<grid_data_t<TYPE, order>, dos> scalar_data_list;

    // MARK:- Vector Data Manipulation
    //! number of vector data at given index.
    CUDA_CALLABLE_DEVICE uint32_t number_of_vector_data() const {
        return scalar_data_list[0].data.shape.size();
    }

    //! Set the scalar data at given index.
    CUDA_CALLABLE_DEVICE void set_vector_data(uint32_t var_idx, const system_array_t &data) {
        for (uint32_t i = 0; i < dos; i++) {
            scalar_data_list[i].data[var_idx] = data[i];
        }
    }

    //! Add the scalar data at given index.
    CUDA_CALLABLE_DEVICE void add_vector_data(uint32_t var_idx, const system_array_t &data) {
        for (uint32_t i = 0; i < dos; i++) {
            scalar_data_list[i].data[var_idx] = scalar_data_list[i].data[var_idx] + data[i];
        }
    }

    //! return value of specific index
    CUDA_CALLABLE_DEVICE system_array_t value(uint32_t var_idx) {
        system_array_t tmp;
        for (uint32_t i = 0; i < dos; i++) {
            tmp[i] = scalar_data_list[i].value(var_idx);
        }
        return tmp;
    }

    //! return value of specific point
    CUDA_CALLABLE_DEVICE system_array_t value(const point_t &pt, uint32_t var_idx) {
        system_array_t tmp;
        for (uint32_t i = 0; i < dos; i++) {
            tmp[i] = scalar_data_list[i].value(pt, var_idx);
        }
        return tmp;
    }

    //! return value of specific point in specific bry
    CUDA_CALLABLE_DEVICE system_array_t value(const point_t &pt, uint32_t var_idx, uint32_t bry) {
        system_array_t tmp;
        for (uint32_t i = 0; i < dos; i++) {
            tmp[i] = scalar_data_list[i].value(pt, var_idx, bry);
        }
        return tmp;
    }

    //! return value of specific point
    CUDA_CALLABLE_DEVICE Vector<Vector<float, dim>, dos> gradient(const point_t &pt, uint32_t var_idx) {
        Vector<Vector<float, dim>, dos> tmp;
        for (uint32_t j = 0; j < dos; j++) {
            Vector<float, dim> g = scalar_data_list[j].gradient(pt, var_idx);
            for (uint32_t k = 0; k < dim; ++k) {
                tmp[j][k] = g[k];
            }
        }
        return tmp;
    }

    //! return value of specific point in specific bry
    CUDA_CALLABLE_DEVICE Vector<Vector<float, dim>, dos> gradient(const point_t &pt, uint32_t var_idx, uint32_t bry) {
        Vector<Vector<float, dim>, dos> tmp;
        for (uint32_t j = 0; j < dos; j++) {
            Vector<float, dim> g = scalar_data_list[j].gradient(pt, var_idx, bry);
            for (uint32_t k = 0; k < dim; ++k) {
                tmp[j][k] = g[k];
            }
        }
        return tmp;
    }
};
}// namespace vox::fields