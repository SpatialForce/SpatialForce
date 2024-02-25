//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "geometry.h"
#include "geometry_trait.h"

namespace vox::fields {
template<typename TYPE>
struct base_template_geometry_t {};

template<typename TYPE, uint32_t ACCURACY>
struct template_geometry_t : public base_template_geometry_t<TYPE> {
    static constexpr uint32_t accuracy = ACCURACY;
};

template<uint32_t DIM, uint32_t SIZE>
struct quadrature_info_t {
    static constexpr uint32_t dim = DIM;
    static constexpr uint32_t size = SIZE;
    /// Algebraic accuracy.
    int32_t alg_acc{};
    /// The coordinate of quadrature point.
    CudaStdArray<Vector<float, dim>, size> pnts;
    /// Quadrature weight on the point.
    CudaStdArray<float, size> weights;
};

}// namespace vox::fields

#include "templates/interval.h"
#include "templates/triangle.h"
#include "templates/tetrahedron.h"