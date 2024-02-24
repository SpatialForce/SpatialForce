//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "tensor.h"
#include "iteration_utils.h"
#include "type_helpers.h"

namespace vox {

template<typename T, size_t N>
void fill(TensorView<T, N> a, const Vector<size_t, N> &begin,
          const Vector<size_t, N> &end, const T &val) {
    forEachIndex(begin, end, [&](auto... idx) { a(idx...) = val; });
}

template<typename T, size_t N>
void fill(TensorView<T, N> a, const T &val) {
    fill(a, Vector<size_t, N>{}, Vector<size_t, N>{a.shape()}, val);
}

template<typename T>
void fill(TensorView<T, 1> a, size_t begin, size_t end, const T &val) {
    fill(a, Vector1UZ{begin}, Vector1UZ{end}, val);
}

template<typename T, typename U, size_t N>
void copy(TensorView<T, N> src, const Vector<size_t, N> &begin,
          const Vector<size_t, N> &end, TensorView<U, N> dst) {
    forEachIndex(begin, end, [&](auto... idx) { dst(idx...) = src(idx...); });
}

template<typename T, typename U, size_t N>
void copy(TensorView<T, N> src, TensorView<U, N> dst) {
    copy(src, Vector<size_t, N>{}, Vector<size_t, N>{src.shape()}, dst);
}

template<typename T, typename U>
void copy(TensorView<T, 1> src, size_t begin, size_t end, TensorView<U, 1> dst) {
    copy(src, Vector1UZ{begin}, Vector1UZ{end}, dst);
}

template<typename T, typename U>
void extrapolateToRegion(TensorView2<T> input, TensorView2<char> valid,
                         unsigned int numberOfIterations,
                         TensorView2<U> output) {
    const Vector2UZ shape = input.shape();

    ASSERT(shape == valid.shape());
    ASSERT(shape == output.shape());

    Tensor2<char> valid0(shape);
    Tensor2<char> valid1(shape);

    forEachIndex(valid0.shape(), [&](size_t i, size_t j) {
        valid0(i, j) = valid(i, j);
        output(i, j) = input(i, j);
    });

    for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
        forEachIndex(valid0.shape(), [&](size_t i, size_t j) {
            T sum = T{};
            unsigned int count = 0;

            if (!valid0(i, j)) {
                if (i + 1 < shape.x && valid0(i + 1, j)) {
                    sum += output(i + 1, j);
                    ++count;
                }

                if (i > 0 && valid0(i - 1, j)) {
                    sum += output(i - 1, j);
                    ++count;
                }

                if (j + 1 < shape.y && valid0(i, j + 1)) {
                    sum += output(i, j + 1);
                    ++count;
                }

                if (j > 0 && valid0(i, j - 1)) {
                    sum += output(i, j - 1);
                    ++count;
                }

                if (count > 0) {
                    output(i, j) =
                        sum /
                        static_cast<typename GetScalarType<T>::value>(count);
                    valid1(i, j) = 1;
                }
            } else {
                valid1(i, j) = 1;
            }
        });

        valid0.swap(valid1);
    }
}

template<typename T, typename U>
void extrapolateToRegion(TensorView3<T> input, TensorView3<char> valid,
                         unsigned int numberOfIterations,
                         TensorView3<U> output) {
    const Vector3UZ shape = input.shape();

    ASSERT(shape == valid.shape());
    ASSERT(shape == output.shape());

    Tensor3<char> valid0(shape);
    Tensor3<char> valid1(shape);

    forEachIndex(valid0.shape(), [&](size_t i, size_t j, size_t k) {
        valid0(i, j, k) = valid(i, j, k);
        output(i, j, k) = input(i, j, k);
    });

    for (unsigned int iter = 0; iter < numberOfIterations; ++iter) {
        forEachIndex(valid0.shape(), [&](size_t i, size_t j, size_t k) {
            T sum = T{};
            unsigned int count = 0;

            if (!valid0(i, j, k)) {
                if (i + 1 < shape.x && valid0(i + 1, j, k)) {
                    sum += output(i + 1, j, k);
                    ++count;
                }

                if (i > 0 && valid0(i - 1, j, k)) {
                    sum += output(i - 1, j, k);
                    ++count;
                }

                if (j + 1 < shape.y && valid0(i, j + 1, k)) {
                    sum += output(i, j + 1, k);
                    ++count;
                }

                if (j > 0 && valid0(i, j - 1, k)) {
                    sum += output(i, j - 1, k);
                    ++count;
                }

                if (k + 1 < shape.z && valid0(i, j, k + 1)) {
                    sum += output(i, j, k + 1);
                    ++count;
                }

                if (k > 0 && valid0(i, j, k - 1)) {
                    sum += output(i, j, k - 1);
                    ++count;
                }

                if (count > 0) {
                    output(i, j, k) =
                        sum /
                        static_cast<typename GetScalarType<T>::value>(count);
                    valid1(i, j, k) = 1;
                }
            } else {
                valid1(i, j, k) = 1;
            }
        });

        valid0.swap(valid1);
    }
}

}// namespace vox