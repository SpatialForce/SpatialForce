//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cmath>
#include <cassert>

namespace vox {

// MARK: Internal Helpers

namespace internal {

// TODO: With C++17, fold expression could be used instead.
template<typename M1, typename M2, size_t J>
struct DotProduct {
    CUDA_CALLABLE constexpr static auto call(const M1 &a, const M2 &b, size_t i, size_t j) {
        return DotProduct<M1, M2, J - 1>::call(a, b, i, j) + a(i, J) * b(J, j);
    }
};

template<typename M1, typename M2>
struct DotProduct<M1, M2, 0> {
    CUDA_CALLABLE constexpr static auto call(const M1 &a, const M2 &b, size_t i, size_t j) {
        return a(i, 0) * b(0, j);
    }
};

// TODO: With C++17, fold expression could be used instead.
template<typename T, size_t Rows, size_t Cols, typename ReduceOperation,
         typename UnaryOperation, size_t I>
struct Reduce {
    // For vector-like Matrix
    template<typename U = T>
    CUDA_CALLABLE constexpr static std::enable_if_t<(Cols == 1), U> call(
        const Matrix<T, Rows, 1> &a, const T &init, ReduceOperation op,
        UnaryOperation uop) {
        return op(
            Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, I - 1>::call(
                a, init, op, uop),
            uop(a(I, 0)));
    }

    // For vector-like Matrix with zero init
    template<typename U = T>
    CUDA_CALLABLE constexpr static std::enable_if_t<(Cols == 1), U> call(
        const Matrix<T, Rows, 1> &a, ReduceOperation op, UnaryOperation uop) {
        return op(
            Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, I - 1>::call(
                a, op, uop),
            uop(a(I, 0)));
    }

    // For Matrix
    CUDA_CALLABLE constexpr static T call(const Matrix<T, Rows, Cols> &a, const T &init,
                                          ReduceOperation op, UnaryOperation uop) {
        return op(
            Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, I - 1>::call(
                a, init, op, uop),
            uop(a[I]));
    }

    // For Matrix with zero init
    CUDA_CALLABLE constexpr static T call(const Matrix<T, Rows, Cols> &a, ReduceOperation op,
                                          UnaryOperation uop) {
        return op(
            Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, I - 1>::call(
                a, op, uop),
            uop(a[I]));
    }

    // For diagonal elements on Matrix
    CUDA_CALLABLE constexpr static T callDiag(const Matrix<T, Rows, Cols> &a, const T &init,
                                              ReduceOperation op, UnaryOperation uop) {
        return op(Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation,
                         I - 1>::callDiag(a, init, op, uop),
                  uop(a(I, I)));
    }
};

template<typename T, size_t Rows, size_t Cols, typename ReduceOperation,
         typename UnaryOperation>
struct Reduce<T, Rows, Cols, ReduceOperation, UnaryOperation, 0> {
    // For vector-like Matrix
    template<typename U = T>
    CUDA_CALLABLE constexpr static std::enable_if_t<(Cols > 1), U> call(
        const Matrix<T, Rows, 1> &a, const T &init, ReduceOperation op,
        UnaryOperation uop) {
        return op(uop(a(0, 0)), init);
    }

    // For vector-like Matrix with zero init
    template<typename U = T>
    CUDA_CALLABLE constexpr static std::enable_if_t<(Cols == 1), U> call(
        const Matrix<T, Rows, 1> &a, ReduceOperation op, UnaryOperation uop) {
        return uop(a(0, 0));
    }

    // For Matrix
    CUDA_CALLABLE constexpr static T call(const Matrix<T, Rows, Cols> &a, const T &init,
                                          ReduceOperation op, UnaryOperation uop) {
        return op(uop(a[0]), init);
    }

    // For MatrixBase with zero init
    CUDA_CALLABLE constexpr static T call(const Matrix<T, Rows, Cols> &a, ReduceOperation op,
                                          UnaryOperation uop) {
        return uop(a[0]);
    }

    // For diagonal elements on MatrixBase
    CUDA_CALLABLE constexpr static T callDiag(const Matrix<T, Rows, Cols> &a, const T &init,
                                              ReduceOperation op, UnaryOperation uop) {
        return op(uop(a(0, 0)), init);
    }
};

// We can use std::logical_and<>, but explicitly putting && helps compiler
// to early terminate the loop (at least for gcc 8.1 as I checked the
// assembly).
// TODO: With C++17, fold expression could be used instead.
template<typename T, size_t Rows, size_t Cols, typename BinaryOperation,
         size_t I>
struct FoldWithAnd {
    CUDA_CALLABLE constexpr static bool call(const Matrix<T, Rows, Cols> &a,
                                             const Matrix<T, Rows, Cols> &b,
                                             BinaryOperation op) {
        return FoldWithAnd<T, Rows, Cols, BinaryOperation, I - 1>::call(a, b,
                                                                        op) &&
               op(a[I], b[I]);
    }
};

template<typename T, size_t Rows, size_t Cols, typename BinaryOperation>
struct FoldWithAnd<T, Rows, Cols, BinaryOperation, 0> {
    CUDA_CALLABLE constexpr static bool call(const Matrix<T, Rows, Cols> &a,
                                             const Matrix<T, Rows, Cols> &b,
                                             BinaryOperation op) {
        return op(a[0], b[0]);
    }
};

}// namespace internal

////////////////////////////////////////////////////////////////////////////////
#pragma region Matrix Class (Static)

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE Matrix<T, Rows, Cols>::Matrix(const_reference value) {
    fill(value);
}

template<typename T, size_t Rows, size_t Cols>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE Matrix<T, Rows, Cols>::Matrix(const MatrixExpression<T, R, C, E> &expression) {
    assert(expression.rows() == Rows && expression.cols() == Cols);

    copyFrom(expression);
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE Matrix<T, Rows, Cols>::Matrix(const NestedInitializerListsT<T, 2> &lst) {
    size_t i = 0;
    for (auto rows : lst) {
        assert(i < Rows);
        size_t j = 0;
        for (auto col : rows) {
            assert(j < Cols);
            (*this)(i, j) = col;
            ++j;
        }
        ++i;
    }
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE Matrix<T, Rows, Cols>::Matrix(const_pointer ptr) {
    size_t cnt = 0;
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols; ++j) {
            (*this)(i, j) = ptr[cnt++];
        }
    }
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE void Matrix<T, Rows, Cols>::fill(const T &val) {
    for (size_t i = 0; i < Rows * Cols; ++i) {
        _elements[i] = val;
    }
}

template<typename T, size_t Rows, size_t Cols>
void Matrix<T, Rows, Cols>::fill(const std::function<T(size_t i)> &func) {
    for (size_t i = 0; i < Rows * Cols; ++i) {
        _elements[i] = func(i);
    }
}

template<typename T, size_t Rows, size_t Cols>
void Matrix<T, Rows, Cols>::fill(
    const std::function<T(size_t i, size_t j)> &func) {
    for (size_t i = 0; i < Rows; ++i) {
        for (size_t j = 0; j < Cols; ++j) {
            (*this)(i, j) = func(i, j);
        }
    }
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE void Matrix<T, Rows, Cols>::swap(Matrix &other) {
    thrust::swap(_elements, other._elements);
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE constexpr size_t Matrix<T, Rows, Cols>::rows() const {
    return Rows;
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE constexpr size_t Matrix<T, Rows, Cols>::cols() const {
    return Cols;
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE typename Matrix<T, Rows, Cols>::iterator Matrix<T, Rows, Cols>::begin() {
    return &_elements[0];
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE constexpr typename Matrix<T, Rows, Cols>::const_iterator
Matrix<T, Rows, Cols>::begin() const {
    return &_elements[0];
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE typename Matrix<T, Rows, Cols>::iterator Matrix<T, Rows, Cols>::end() {
    return begin() + Rows * Cols;
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE constexpr typename Matrix<T, Rows, Cols>::const_iterator
Matrix<T, Rows, Cols>::end() const {
    return begin() + Rows * Cols;
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE typename Matrix<T, Rows, Cols>::pointer Matrix<T, Rows, Cols>::data() {
    return &_elements[0];
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE constexpr typename Matrix<T, Rows, Cols>::const_pointer
Matrix<T, Rows, Cols>::data() const {
    return &_elements[0];
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE typename Matrix<T, Rows, Cols>::reference Matrix<T, Rows, Cols>::operator[](
    size_t i) {
    assert(i < Rows * Cols);
    return _elements[i];
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE typename Matrix<T, Rows, Cols>::const_reference Matrix<T, Rows, Cols>::
operator[](size_t i) const {
    assert(i < Rows * Cols);
    return _elements[i];
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region Matrix<T, 1, 1> (aka Vector1)

template<typename T>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE Matrix<T, 1, 1>::Matrix(const MatrixExpression<T, R, C, E> &expression) {
    assert(expression.rows() == 1 && expression.cols() == 1);

    x = expression.eval(0, 0);
}

template<typename T>
CUDA_CALLABLE Matrix<T, 1, 1>::Matrix(const std::initializer_list<T> &lst) {
    assert(lst.size() > 0);

    x = *lst.begin();
}

template<typename T>
CUDA_CALLABLE void Matrix<T, 1, 1>::fill(const T &val) {
    x = val;
}

template<typename T>
void Matrix<T, 1, 1>::fill(const std::function<T(size_t i)> &func) {
    x = func(0);
}

template<typename T>
void Matrix<T, 1, 1>::fill(const std::function<T(size_t i, size_t j)> &func) {
    x = func(0, 0);
}

template<typename T>
CUDA_CALLABLE void Matrix<T, 1, 1>::swap(Matrix &other) {
    thrust::swap(x, other.x);
}

template<typename T>
CUDA_CALLABLE constexpr size_t Matrix<T, 1, 1>::rows() const {
    return 1;
}

template<typename T>
CUDA_CALLABLE constexpr size_t Matrix<T, 1, 1>::cols() const {
    return 1;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 1, 1>::iterator Matrix<T, 1, 1>::begin() {
    return &x;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 1, 1>::const_iterator Matrix<T, 1, 1>::begin()
    const {
    return &x;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 1, 1>::iterator Matrix<T, 1, 1>::end() {
    return begin() + 1;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 1, 1>::const_iterator Matrix<T, 1, 1>::end()
    const {
    return begin() + 1;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 1, 1>::pointer Matrix<T, 1, 1>::data() {
    return &x;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 1, 1>::const_pointer Matrix<T, 1, 1>::data()
    const {
    return &x;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 1, 1>::reference Matrix<T, 1, 1>::operator[](size_t i) {
    assert(i < 1);
    return (&x)[i];
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 1, 1>::const_reference Matrix<T, 1, 1>::operator[](
    size_t i) const {
    assert(i < 1);
    return (&x)[i];
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 1, 1> Matrix<T, 1, 1>::makeUnitX() {
    return Matrix<T, 1, 1>(1);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 1, 1> Matrix<T, 1, 1>::makeUnit(size_t) {
    return makeUnitX();
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region Matrix<T, 2, 1> (aka Vector2)

template<typename T>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE Matrix<T, 2, 1>::Matrix(const MatrixExpression<T, R, C, E> &expression) {
    assert(expression.rows() == 2 && expression.cols() == 1);

    x = expression.eval(0, 0);
    y = expression.eval(1, 0);
}

template<typename T>
CUDA_CALLABLE Matrix<T, 2, 1>::Matrix(const std::initializer_list<T> &lst) {
    assert(lst.size() > 1);

    auto iter = lst.begin();
    x = *(iter++);
    y = *(iter);
}

template<typename T>
CUDA_CALLABLE void Matrix<T, 2, 1>::fill(const T &val) {
    x = y = val;
}

template<typename T>
void Matrix<T, 2, 1>::fill(const std::function<T(size_t i)> &func) {
    x = func(0);
    y = func(1);
}

template<typename T>
void Matrix<T, 2, 1>::fill(const std::function<T(size_t i, size_t j)> &func) {
    x = func(0, 0);
    y = func(1, 0);
}

template<typename T>
CUDA_CALLABLE void Matrix<T, 2, 1>::swap(Matrix &other) {
    thrust::swap(x, other.x);
    thrust::swap(y, other.y);
}

template<typename T>
CUDA_CALLABLE constexpr size_t Matrix<T, 2, 1>::rows() const {
    return 2;
}

template<typename T>
CUDA_CALLABLE constexpr size_t Matrix<T, 2, 1>::cols() const {
    return 1;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 2, 1>::iterator Matrix<T, 2, 1>::begin() {
    return &x;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 2, 1>::const_iterator Matrix<T, 2, 1>::begin()
    const {
    return &x;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 2, 1>::iterator Matrix<T, 2, 1>::end() {
    return begin() + 2;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 2, 1>::const_iterator Matrix<T, 2, 1>::end()
    const {
    return begin() + 2;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 2, 1>::pointer Matrix<T, 2, 1>::data() {
    return &x;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 2, 1>::const_pointer Matrix<T, 2, 1>::data()
    const {
    return &x;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 2, 1>::reference Matrix<T, 2, 1>::operator[](size_t i) {
    assert(i < 2);
    return (&x)[i];
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 2, 1>::const_reference Matrix<T, 2, 1>::operator[](
    size_t i) const {
    assert(i < 2);
    return (&x)[i];
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 2, 1> Matrix<T, 2, 1>::makeUnitX() {
    return Matrix<T, 2, 1>(1, 0);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 2, 1> Matrix<T, 2, 1>::makeUnitY() {
    return Matrix<T, 2, 1>(0, 1);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 2, 1> Matrix<T, 2, 1>::makeUnit(size_t i) {
    return Matrix<T, 2, 1>(i == 0, i == 1);
}

#pragma endregion
////////////////////////////////////////////////////////////////////////////////
#pragma region Matrix<T, 3, 1> (aka Vector3)

template<typename T>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE Matrix<T, 3, 1>::Matrix(const MatrixExpression<T, R, C, E> &expression) {
    assert(expression.rows() == 3 && expression.cols() == 1);

    x = expression.eval(0, 0);
    y = expression.eval(1, 0);
    z = expression.eval(2, 0);
}

template<typename T>
CUDA_CALLABLE Matrix<T, 3, 1>::Matrix(const std::initializer_list<T> &lst) {
    assert(lst.size() > 2);

    auto iter = lst.begin();
    x = *(iter++);
    y = *(iter++);
    z = *(iter);
}

template<typename T>
CUDA_CALLABLE void Matrix<T, 3, 1>::fill(const T &val) {
    x = y = z = val;
}

template<typename T>
void Matrix<T, 3, 1>::fill(const std::function<T(size_t i)> &func) {
    x = func(0);
    y = func(1);
    z = func(2);
}

template<typename T>
void Matrix<T, 3, 1>::fill(const std::function<T(size_t i, size_t j)> &func) {
    x = func(0, 0);
    y = func(1, 0);
    z = func(2, 0);
}

template<typename T>
CUDA_CALLABLE void Matrix<T, 3, 1>::swap(Matrix &other) {
    thrust::swap(x, other.x);
    thrust::swap(y, other.y);
    thrust::swap(z, other.z);
}

template<typename T>
CUDA_CALLABLE constexpr size_t Matrix<T, 3, 1>::rows() const {
    return 3;
}

template<typename T>
CUDA_CALLABLE constexpr size_t Matrix<T, 3, 1>::cols() const {
    return 1;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 3, 1>::iterator Matrix<T, 3, 1>::begin() {
    return &x;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 3, 1>::const_iterator Matrix<T, 3, 1>::begin()
    const {
    return &x;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 3, 1>::iterator Matrix<T, 3, 1>::end() {
    return begin() + 3;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 3, 1>::const_iterator Matrix<T, 3, 1>::end()
    const {
    return begin() + 3;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 3, 1>::pointer Matrix<T, 3, 1>::data() {
    return &x;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 3, 1>::const_pointer Matrix<T, 3, 1>::data()
    const {
    return &x;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 3, 1>::reference Matrix<T, 3, 1>::operator[](size_t i) {
    assert(i < 3);
    return (&x)[i];
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 3, 1>::const_reference Matrix<T, 3, 1>::operator[](
    size_t i) const {
    assert(i < 3);
    return (&x)[i];
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::makeUnitX() {
    return Matrix<T, 3, 1>(1, 0, 0);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::makeUnitY() {
    return Matrix<T, 3, 1>(0, 1, 0);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::makeUnitZ() {
    return Matrix<T, 3, 1>(0, 0, 1);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 3, 1> Matrix<T, 3, 1>::makeUnit(size_t i) {
    return Matrix<T, 3, 1>(i == 0, i == 1, i == 2);
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region Matrix<T, 4, 1> (aka Vector4)

template<typename T>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE Matrix<T, 4, 1>::Matrix(const MatrixExpression<T, R, C, E> &expression) {
    assert(expression.rows() == 4 && expression.cols() == 1);

    x = expression.eval(0, 0);
    y = expression.eval(1, 0);
    z = expression.eval(2, 0);
    w = expression.eval(3, 0);
}

template<typename T>
CUDA_CALLABLE Matrix<T, 4, 1>::Matrix(const std::initializer_list<T> &lst) {
    assert(lst.size() > 3);

    auto iter = lst.begin();
    x = *(iter++);
    y = *(iter++);
    z = *(iter++);
    w = *(iter);
}

template<typename T>
CUDA_CALLABLE void Matrix<T, 4, 1>::fill(const T &val) {
    x = y = z = w = val;
}

template<typename T>
void Matrix<T, 4, 1>::fill(const std::function<T(size_t i)> &func) {
    x = func(0);
    y = func(1);
    z = func(2);
    w = func(3);
}

template<typename T>
void Matrix<T, 4, 1>::fill(const std::function<T(size_t i, size_t j)> &func) {
    x = func(0, 0);
    y = func(1, 0);
    z = func(2, 0);
    w = func(3, 0);
}

template<typename T>
CUDA_CALLABLE void Matrix<T, 4, 1>::swap(Matrix &other) {
    thrust::swap(x, other.x);
    thrust::swap(y, other.y);
    thrust::swap(z, other.z);
    thrust::swap(w, other.w);
}

template<typename T>
CUDA_CALLABLE constexpr size_t Matrix<T, 4, 1>::rows() const {
    return 4;
}

template<typename T>
CUDA_CALLABLE constexpr size_t Matrix<T, 4, 1>::cols() const {
    return 1;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 4, 1>::iterator Matrix<T, 4, 1>::begin() {
    return &x;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 4, 1>::const_iterator Matrix<T, 4, 1>::begin()
    const {
    return &x;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 4, 1>::iterator Matrix<T, 4, 1>::end() {
    return begin() + 4;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 4, 1>::const_iterator Matrix<T, 4, 1>::end()
    const {
    return begin() + 4;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 4, 1>::pointer Matrix<T, 4, 1>::data() {
    return &x;
}

template<typename T>
CUDA_CALLABLE constexpr typename Matrix<T, 4, 1>::const_pointer Matrix<T, 4, 1>::data()
    const {
    return &x;
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 4, 1>::reference Matrix<T, 4, 1>::operator[](size_t i) {
    assert(i < 4);
    return (&x)[i];
}

template<typename T>
CUDA_CALLABLE typename Matrix<T, 4, 1>::const_reference Matrix<T, 4, 1>::operator[](
    size_t i) const {
    assert(i < 4);
    return (&x)[i];
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 4, 1> Matrix<T, 4, 1>::makeUnitX() {
    return Matrix<T, 4, 1>(1, 0, 0, 0);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 4, 1> Matrix<T, 4, 1>::makeUnitY() {
    return Matrix<T, 4, 1>(0, 1, 0, 0);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 4, 1> Matrix<T, 4, 1>::makeUnitZ() {
    return Matrix<T, 4, 1>(0, 0, 1, 0);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 4, 1> Matrix<T, 4, 1>::makeUnitW() {
    return Matrix<T, 4, 1>(0, 0, 0, 1);
}

template<typename T>
CUDA_CALLABLE constexpr Matrix<T, 4, 1> Matrix<T, 4, 1>::makeUnit(size_t i) {
    return Matrix<T, 4, 1>(i == 0, i == 1, i == 2, i == 3);
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region Matrix Class (Dynamic)

template<typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix() {}

template<typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(
    size_t rows, size_t cols, const_reference value) {
    _elements.resize(rows * cols);
    _rows = rows;
    _cols = cols;
    fill(value);
}

template<typename T>
template<size_t R, size_t C, typename E>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(
    const MatrixExpression<T, R, C, E> &expression)
    : Matrix(expression.rows(), expression.cols()) {
    copyFrom(expression);
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(
    const NestedInitializerListsT<T, 2> &lst) {
    size_t i = 0;
    for (auto rows : lst) {
        size_t j = 0;
        for (auto col : rows) {
            (void)col;
            ++j;
        }
        _cols = j;
        ++i;
    }
    _rows = i;
    _elements.resize(_rows * _cols);

    i = 0;
    for (auto rows : lst) {
        assert(i < _rows);
        size_t j = 0;
        for (auto col : rows) {
            assert(j < _cols);
            (*this)(i, j) = col;
            ++j;
        }
        ++i;
    }
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(size_t rows,
                                                          size_t cols,
                                                          const_pointer ptr)
    : Matrix(rows, cols) {
    size_t cnt = 0;
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            (*this)(i, j) = ptr[cnt++];
        }
    }
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(const Matrix &other)
    : _elements(other._elements), _rows(other._rows), _cols(other._cols) {}

template<typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::Matrix(Matrix &&other) {
    *this = std::move(other);
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::fill(const T &val) {
    std::fill(_elements.begin(), _elements.end(), val);
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::fill(
    const std::function<T(size_t i)> &func) {
    for (size_t i = 0; i < _elements.size(); ++i) {
        _elements[i] = func(i);
    }
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::fill(
    const std::function<T(size_t i, size_t j)> &func) {
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            (*this)(i, j) = func(i, j);
        }
    }
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::swap(Matrix &other) {
    _elements.swap(other._elements);
    std::swap(_rows, other._rows);
    std::swap(_cols, other._cols);
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::resize(
    size_t rows, size_t cols, const_reference val) {
    Matrix newMatrix{rows, cols, val};
    size_t minRows = std::min(rows, _rows);
    size_t minCols = std::min(cols, _cols);
    for (size_t i = 0; i < minRows; ++i) {
        for (size_t j = 0; j < minCols; ++j) {
            newMatrix(i, j) = (*this)(i, j);
        }
    }
    *this = std::move(newMatrix);
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::clear() {
    _elements.clear();
    _rows = 0;
    _cols = 0;
}

template<typename T>
size_t Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::rows() const {
    return _rows;
}

template<typename T>
size_t Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::cols() const {
    return _cols;
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::iterator
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::begin() {
    return &_elements[0];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::const_iterator
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::begin() const {
    return &_elements[0];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::iterator
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::end() {
    return begin() + _rows * _cols;
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::const_iterator
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::end() const {
    return begin() + _rows * _cols;
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::pointer
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::data() {
    return &_elements[0];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::const_pointer
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::data() const {
    return &_elements[0];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::reference
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::operator[](size_t i) {
    assert(i < _rows * _cols);
    return _elements[i];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::const_reference
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::operator[](
    size_t i) const {
    assert(i < _rows * _cols);
    return _elements[i];
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic> &
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::operator=(
    const Matrix &other) {
    _elements = other._elements;
    _rows = other._rows;
    _cols = other._cols;
    return *this;
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic> &
Matrix<T, kMatrixSizeDynamic, kMatrixSizeDynamic>::operator=(Matrix &&other) {
    _elements = std::move(other._elements);
    _rows = other._rows;
    _cols = other._cols;
    other._rows = 0;
    other._cols = 0;
    return *this;
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region Specialized Matrix for Dynamic Vector Type

template<typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix() {}

template<typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(size_t rows, const_reference value) {
    _elements.resize(rows, value);
}

template<typename T>
template<size_t R, size_t C, typename E>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(
    const MatrixExpression<T, R, C, E> &expression)
    : Matrix(expression.rows(), 1) {
    copyFrom(expression);
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(const std::initializer_list<T> &lst) {
    size_t sz = lst.size();
    _elements.resize(sz);

    size_t i = 0;
    for (auto row : lst) {
        _elements[i] = static_cast<T>(row);
        ++i;
    }
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(size_t rows, const_pointer ptr)
    : Matrix(rows) {
    size_t cnt = 0;
    for (size_t i = 0; i < rows; ++i) {
        (*this)[i] = ptr[cnt++];
    }
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(const Matrix &other)
    : _elements(other._elements) {}

template<typename T>
Matrix<T, kMatrixSizeDynamic, 1>::Matrix(Matrix &&other) {
    *this = std::move(other);
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::fill(const T &val) {
    std::fill(_elements.begin(), _elements.end(), val);
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::fill(
    const std::function<T(size_t i)> &func) {
    for (size_t i = 0; i < _elements.size(); ++i) {
        _elements[i] = func(i);
    }
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::fill(
    const std::function<T(size_t i, size_t j)> &func) {
    for (size_t i = 0; i < rows(); ++i) {
        _elements[i] = func(i, 0);
    }
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::swap(Matrix &other) {
    _elements.swap(other._elements);
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::resize(size_t rows,
                                              const_reference val) {
    _elements.resize(rows, val);
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::addElement(const_reference newElem) {
    _elements.push_back(newElem);
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::addElement(const Matrix &newElems) {
    _elements.insert(_elements.end(), newElems._elements.begin(),
                     newElems._elements.end());
}

template<typename T>
void Matrix<T, kMatrixSizeDynamic, 1>::clear() {
    _elements.clear();
}

template<typename T>
size_t Matrix<T, kMatrixSizeDynamic, 1>::rows() const {
    return _elements.size();
}

template<typename T>
constexpr size_t Matrix<T, kMatrixSizeDynamic, 1>::cols() const {
    return 1;
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::iterator
Matrix<T, kMatrixSizeDynamic, 1>::begin() {
    return &_elements[0];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::const_iterator
Matrix<T, kMatrixSizeDynamic, 1>::begin() const {
    return &_elements[0];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::iterator
Matrix<T, kMatrixSizeDynamic, 1>::end() {
    return begin() + _elements.size();
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::const_iterator
Matrix<T, kMatrixSizeDynamic, 1>::end() const {
    return begin() + _elements.size();
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::pointer
Matrix<T, kMatrixSizeDynamic, 1>::data() {
    return &_elements[0];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::const_pointer
Matrix<T, kMatrixSizeDynamic, 1>::data() const {
    return &_elements[0];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::reference
Matrix<T, kMatrixSizeDynamic, 1>::operator[](size_t i) {
    assert(i < _elements.size());
    return _elements[i];
}

template<typename T>
typename Matrix<T, kMatrixSizeDynamic, 1>::const_reference
Matrix<T, kMatrixSizeDynamic, 1>::operator[](size_t i) const {
    assert(i < _elements.size());
    return _elements[i];
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, 1> &Matrix<T, kMatrixSizeDynamic, 1>::operator=(
    const Matrix &other) {
    _elements = other._elements;
    return *this;
}

template<typename T>
Matrix<T, kMatrixSizeDynamic, 1> &Matrix<T, kMatrixSizeDynamic, 1>::operator=(
    Matrix &&other) {
    _elements = std::move(other._elements);
    return *this;
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
// MARK: Matrix Operators

#pragma region Binary Operators

// *

template<typename T, size_t Rows>
[[deprecated("Use elemMul instead")]] CUDA_CALLABLE constexpr auto operator*(
    const Vector<T, Rows> &a, const Vector<T, Rows> &b) {
    return MatrixElemWiseMul<T, Rows, 1, const Vector<T, Rows> &,
                             const Vector<T, Rows> &>{a, b};
}

// /

template<typename T, size_t Rows>
[[deprecated("Use elemDiv instead")]] CUDA_CALLABLE constexpr auto operator/(
    const Vector<T, Rows> &a, const Vector<T, Rows> &b) {
    return MatrixElemWiseDiv<T, Rows, 1, const Vector<T, Rows> &,
                             const Vector<T, Rows> &>{a, b.derived()};
}

#pragma endregion

#pragma region Assignment Operators

// +=

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
CUDA_CALLABLE void operator+=(Matrix<T, R1, C1> &a,
                              const MatrixExpression<T, R2, C2, M2> &b) {
    a = a + b;
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE void operator+=(Matrix<T, Rows, Cols> &a, const T &b) {
    a = a + b;
}

// -=

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
CUDA_CALLABLE void operator-=(Matrix<T, R1, C1> &a,
                              const MatrixExpression<T, R2, C2, M2> &b) {
    a = a - b;
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE void operator-=(Matrix<T, Rows, Cols> &a, const T &b) {
    a = a - b;
}

// *=

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
CUDA_CALLABLE void operator*=(Matrix<T, R1, C1> &a,
                              const MatrixExpression<T, R2, C2, M2> &b) {
    assert(a.cols() == b.rows());

    Matrix<T, R1, C2> c = a * b;
    a = c;
}

template<typename T, size_t R1, size_t R2, size_t C2, typename M2>
[[deprecated("Use elemIMul instead")]] CUDA_CALLABLE void operator*=(
    Matrix<T, R1, 1> &a, const MatrixExpression<T, R2, C2, M2> &b) {
    assert(a.rows() == b.rows() && a.cols() == b.cols());

    a = MatrixElemWiseMul<T, R1, 1, const Matrix<T, R1, 1> &, const M2 &>{
        a, b.derived()};
}

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
CUDA_CALLABLE void elemIMul(Matrix<T, R1, C1> &a, const MatrixExpression<T, R2, C2, M2> &b) {
    assert(a.rows() == b.rows() && a.cols() == b.cols());

    a = MatrixElemWiseMul<T, R1, C1, const Matrix<T, R1, C1> &, const M2 &>{
        a, b.derived()};
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE void operator*=(Matrix<T, Rows, Cols> &a, const T &b) {
    a = MatrixScalarElemWiseMul<T, Rows, Cols, const Matrix<T, Rows, Cols> &>{a,
                                                                              b};
}

// /=

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
[[deprecated("Use elemIDiv instead")]] CUDA_CALLABLE void operator/=(
    Matrix<T, R1, C1> &a, const MatrixExpression<T, R2, C2, M2> &b) {
    a = MatrixElemWiseDiv<T, R1, C1, const Matrix<T, R1, C1> &, const M2 &>(
        a, b.derived());
}

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M2>
CUDA_CALLABLE void elemIDiv(Matrix<T, R1, C1> &a, const MatrixExpression<T, R2, C2, M2> &b) {
    a = MatrixElemWiseDiv<T, R1, C1, const Matrix<T, R1, C1> &, const M2 &>(
        a, b.derived());
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE void operator/=(Matrix<T, Rows, Cols> &a, const T &b) {
    a = MatrixScalarElemWiseDiv<T, Rows, Cols, const Matrix<T, Rows, Cols> &>{a,
                                                                              b};
}

#pragma endregion

#pragma region Comparison Operators

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), bool> operator==(
    const MatrixExpression<T, Rows, Cols, M1> &a,
    const MatrixExpression<T, Rows, Cols, M2> &b) {
    return internal::FoldWithAnd<T, Rows, Cols, thrust::equal_to<T>,
                                 Rows * Cols - 1>::call(a, b,
                                                        thrust::equal_to<T>());
}

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M1,
         typename M2>
CUDA_CALLABLE bool operator==(const MatrixExpression<T, R1, C1, M1> &a,
                              const MatrixExpression<T, R2, C2, M2> &b) {
    if (a.rows() != b.rows() || a.cols() != b.cols()) {
        return false;
    }

    for (size_t i = 0; i < a.rows(); ++i) {
        for (size_t j = 0; j < a.cols(); ++j) {
            if (a.eval(i, j) != b.eval(i, j)) {
                return false;
            }
        }
    }
    return true;
}

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M1,
         typename M2>
CUDA_CALLABLE bool operator!=(const MatrixExpression<T, R1, C1, M1> &a,
                              const MatrixExpression<T, R2, C2, M2> &b) {
    return !(a == b);
}

#pragma endregion

#pragma region Simple Utilities

// Static Accumulate

template<typename T, size_t Rows, size_t Cols, typename M1,
         typename BinaryOperation>
CUDA_CALLABLE constexpr std::enable_if_t<IsMatrixSizeStatic<Rows, Cols>::value, T> accumulate(
    const MatrixExpression<T, Rows, Cols, M1> &a, const T &init,
    BinaryOperation op) {
    return internal::Reduce<T, Rows, Cols, BinaryOperation, NoOp<T>,
                            Rows * Cols - 1>::call(a, init, op, NoOp<T>());
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr std::enable_if_t<IsMatrixSizeStatic<Rows, Cols>::value, T> accumulate(
    const MatrixExpression<T, Rows, Cols, M1> &a, const T &init) {
    return internal::Reduce<T, Rows, Cols, thrust::plus<T>, NoOp<T>,
                            Rows * Cols - 1>::call(a, init, thrust::plus<T>(),
                                                   NoOp<T>());
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr std::enable_if_t<IsMatrixSizeStatic<Rows, Cols>::value, T> accumulate(
    const MatrixExpression<T, Rows, Cols, M1> &a) {
    return internal::Reduce<T, Rows, Cols, thrust::plus<T>, NoOp<T>,
                            Rows * Cols - 1>::call(a, thrust::plus<T>(),
                                                   NoOp<T>());
}

// Dynamic Accumulate

template<typename T, size_t Rows, size_t Cols, typename M1,
         typename BinaryOperation>
CUDA_CALLABLE constexpr std::enable_if_t<IsMatrixSizeDynamic<Rows, Cols>::value, T>
accumulate(const MatrixExpression<T, Rows, Cols, M1> &a, const T &init,
           BinaryOperation op) {
    return std::accumulate(a.begin(), a.end(), init, op);
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr std::enable_if_t<IsMatrixSizeDynamic<Rows, Cols>::value, T>
accumulate(const MatrixExpression<T, Rows, Cols, M1> &a, const T &init) {
    return std::accumulate(a.begin(), a.end(), init, thrust::plus<T>());
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr std::enable_if_t<IsMatrixSizeDynamic<Rows, Cols>::value, T>
accumulate(const MatrixExpression<T, Rows, Cols, M1> &a) {
    return std::accumulate(a.begin(), a.end(), T{}, thrust::plus<T>());
}

// Product

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr T product(const MatrixExpression<T, Rows, Cols, M1> &a,
                                  const T &init) {
    return accumulate(a, init, thrust::multiplies<T>());
}

// Interpolation
template<typename T, size_t Rows, size_t Cols, typename M1, typename M2,
         typename M3, typename M4>
CUDA_CALLABLE std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), Matrix<T, Rows, Cols>>
monotonicCatmullRom(const MatrixExpression<T, Rows, Cols, M1> &f0,
                    const MatrixExpression<T, Rows, Cols, M2> &f1,
                    const MatrixExpression<T, Rows, Cols, M3> &f2,
                    const MatrixExpression<T, Rows, Cols, M4> &f3, T f) {
    Matrix<T, Rows, Cols> result;
    for (size_t i = 0; i < f0.rows(); ++i) {
        for (size_t j = 0; j < f0.cols(); ++j) {
            result(i, j) = monotonicCatmullRom(f0.eval(i, j), f1.eval(i, j),
                                               f2.eval(i, j), f3.eval(i, j), f);
        }
    }

    return result;
}

#pragma endregion

}// namespace vox