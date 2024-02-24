//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "tensor.h"
#include "tensor_view.h"

namespace vox {

template<typename T, size_t N>
void fill(TensorView<T, N> a, const Vector<size_t, N> &begin,
          const Vector<size_t, N> &end, const T &val);

template<typename T, size_t N>
void fill(TensorView<T, N> a, const T &val);

template<typename T>
void fill(TensorView<T, 1> a, size_t begin, size_t end, const T &val);

template<typename T, typename U, size_t N>
void copy(TensorView<T, N> src, const Vector<size_t, N> &begin,
          const Vector<size_t, N> &end, TensorView<U, N> dst);

template<typename T, typename U, size_t N>
void copy(TensorView<T, N> src, TensorView<U, N> dst);

template<typename T, typename U>
void copy(TensorView<T, 1> src, size_t begin, size_t end, TensorView<U, 1> dst);

//!
//! \brief Extrapolates 2-D input data from 'valid' (1) to 'invalid' (0) region.
//!
//! This function extrapolates 2-D input data from 'valid' (1) to 'invalid' (0)
//! region. It iterates multiple times to propagate the 'valid' values to nearby
//! 'invalid' region. The maximum distance of the propagation is equal to
//! numberOfIterations. The input parameters 'valid' and 'data' should be
//! collocated.
//!
//! \param input - data to extrapolate
//! \param valid - set 1 if valid, else 0.
//! \param numberOfIterations - number of iterations for propagation
//! \param output - extrapolated output
//!
template<typename T, typename U>
void extrapolateToRegion(TensorView2<T> input, TensorView2<char> valid,
                         unsigned int numberOfIterations, TensorView2<U> output);

//!
//! \brief Extrapolates 3-D input data from 'valid' (1) to 'invalid' (0) region.
//!
//! This function extrapolates 3-D input data from 'valid' (1) to 'invalid' (0)
//! region. It iterates multiple times to propagate the 'valid' values to nearby
//! 'invalid' region. The maximum distance of the propagation is equal to
//! numberOfIterations. The input parameters 'valid' and 'data' should be
//! collocated.
//!
//! \param input - data to extrapolate
//! \param valid - set 1 if valid, else 0.
//! \param numberOfIterations - number of iterations for propagation
//! \param output - extrapolated output
//!
template<typename T, typename U>
void extrapolateToRegion(TensorView3<T> input, TensorView3<char> valid,
                         unsigned int numberOfIterations, TensorView3<U> output);

}// namespace vox

#include "tensor_utils-inl.h"