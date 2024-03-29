//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "cuda_std_array.h"

namespace vox {

////////////////////////////////////////////////////////////////////////////////
#pragma region Simple Setters/Modifiers

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE void MatrixDenseBase<T, Rows, Cols, D>::copyFrom(
    const MatrixExpression<T, R, C, E> &expression) {
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            (*this)(i, j) = expression.eval(i, j);
        }
    }
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE void MatrixDenseBase<T, Rows, Cols, D>::setDiagonal(const_reference val) {
    size_t n = ::min(rows(), cols());
    for (size_t i = 0; i < n; ++i) {
        (*this)(i, i) = val;
    }
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE void MatrixDenseBase<T, Rows, Cols, D>::setOffDiagonal(const_reference val) {
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            if (i != j) {
                (*this)(i, j) = val;
            }
        }
    }
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE void MatrixDenseBase<T, Rows, Cols, D>::setRow(
    size_t i, const MatrixExpression<T, R, C, E> &row) {
    assert(row.rows() == cols() && row.cols() == 1);
    for (size_t j = 0; j < cols(); ++j) {
        (*this)(i, j) = row.eval(j, 0);
    }
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE void MatrixDenseBase<T, Rows, Cols, D>::setColumn(
    size_t j, const MatrixExpression<T, R, C, E> &col) {
    assert(col.rows() == rows() && col.cols() == 1);
    for (size_t i = 0; i < rows(); ++i) {
        (*this)(i, j) = col.eval(i, 0);
    }
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE void MatrixDenseBase<T, Rows, Cols, D>::normalize() {
    derived() /= derived().norm();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE void MatrixDenseBase<T, Rows, Cols, D>::transpose() {
    D tmp = derived().transposed();
    copyFrom(tmp);
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE void MatrixDenseBase<T, Rows, Cols, D>::invert() {
    copyFrom(derived().inverse());
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region Operator Overloadings

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE typename MatrixDenseBase<T, Rows, Cols, D>::reference
MatrixDenseBase<T, Rows, Cols, D>::operator()(size_t i, size_t j) {
    assert(i < rows() && j < cols());
    return derived()[j + i * cols()];
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE typename MatrixDenseBase<T, Rows, Cols, D>::const_reference
MatrixDenseBase<T, Rows, Cols, D>::operator()(size_t i, size_t j) const {
    assert(i < rows() && j < cols());
    return derived()[j + i * cols()];
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE MatrixDenseBase<T, Rows, Cols, D> &MatrixDenseBase<T, Rows, Cols, D>::operator=(
    const MatrixExpression<T, R, C, E> &expression) {
    copyFrom(expression);
    return *this;
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region Builders

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeZero() {
    return MatrixConstant<T, Rows, Cols>{Rows, Cols, 0};
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeZero(size_t rows, size_t cols) {
    return MatrixConstant<T, Rows, Cols>{rows, cols, 0};
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeConstant(value_type val) {
    return MatrixConstant<T, Rows, Cols>{Rows, Cols, val};
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeConstant(size_t rows, size_t cols,
                                                      value_type val) {
    return MatrixConstant<T, Rows, Cols>{rows, cols, val};
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeIdentity() {
    using ConstType = MatrixConstant<T, Rows, Cols>;
    return MatrixDiagonal<T, Rows, Cols, ConstType>{ConstType{Rows, Cols, 1}};
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeIdentity(size_t rows) {
    using ConstType = MatrixConstant<T, Rows, Cols>;
    return MatrixDiagonal<T, Rows, Cols, ConstType>{ConstType{rows, rows, 1}};
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<typename... Args, typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeScaleMatrix(value_type first,
                                                         Args... rest) {
    static_assert(sizeof...(rest) == Rows - 1,
                  "Number of parameters should match the size of diagonal.");
    D m{};
    CudaStdArray<T, Rows> diag{{first, rest...}};
    for (size_t i = 0; i < Rows; ++i) {
        m(i, i) = diag[i];
    }
    return m;
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<size_t R, size_t C, typename E, typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeScaleMatrix(
    const MatrixExpression<T, R, C, E> &expression) {
    assert(expression.cols() == 1);
    D m{};
    for (size_t i = 0; i < Rows; ++i) {
        m(i, i) = expression.eval(i, 0);
    }
    return m;
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 2), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeRotationMatrix(T rad) {
    return D{::cos(rad), -::sin(rad), ::sin(rad), ::cos(rad)};
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<size_t R, size_t C, typename E, typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 3 || Rows == 4), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeRotationMatrix(
    const MatrixExpression<T, R, C, E> &axis, T rad) {
    assert(axis.rows() == 3 && axis.cols() == 1);

    D result = makeIdentity();

    result(0, 0) =
        1 + (1 - std::cos(rad)) * (axis.eval(0, 0) * axis.eval(0, 0) - 1);
    result(0, 1) = -axis.eval(2, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis.eval(0, 0) * axis.eval(1, 0);
    result(0, 2) = axis.eval(1, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis.eval(0, 0) * axis.eval(2, 0);

    result(1, 0) = axis.eval(2, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis.eval(0, 0) * axis.eval(1, 0);
    result(1, 1) =
        1 + (1 - std::cos(rad)) * (axis.eval(1, 0) * axis.eval(1, 0) - 1);
    result(1, 2) = -axis.eval(0, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis.eval(1, 0) * axis.eval(2, 0);

    result(2, 0) = -axis.eval(1, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis.eval(0, 0) * axis.eval(2, 0);
    result(2, 1) = axis.eval(0, 0) * std::sin(rad) +
                   (1 - std::cos(rad)) * axis.eval(1, 0) * axis.eval(2, 0);
    result(2, 2) =
        1 + (1 - std::cos(rad)) * (axis.eval(2, 0) * axis.eval(2, 0) - 1);

    return result;
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<size_t R, size_t C, typename E, typename D>
CUDA_CALLABLE std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 4), D>
MatrixDenseBase<T, Rows, Cols, Derived>::makeTranslationMatrix(
    const MatrixExpression<T, R, C, E> &t) {
    assert(t.rows() == 3 && t.cols() == 1);

    D result = makeIdentity();
    result(0, 3) = t.eval(0, 0);
    result(1, 3) = t.eval(1, 0);
    result(2, 3) = t.eval(2, 0);

    return result;
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region Private Helpers

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr size_t MatrixDenseBase<T, Rows, Cols, D>::rows() const {
    return static_cast<const D &>(*this).rows();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr size_t MatrixDenseBase<T, Rows, Cols, D>::cols() const {
    return static_cast<const D &>(*this).cols();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE auto MatrixDenseBase<T, Rows, Cols, D>::begin() {
    return derived().begin();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr auto MatrixDenseBase<T, Rows, Cols, D>::begin() const {
    return derived().begin();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE auto MatrixDenseBase<T, Rows, Cols, D>::end() {
    return derived().end();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr auto MatrixDenseBase<T, Rows, Cols, D>::end() const {
    return derived().end();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE typename MatrixDenseBase<T, Rows, Cols, D>::reference
MatrixDenseBase<T, Rows, Cols, D>::operator[](size_t i) {
    assert(i < rows() * cols());
    return derived()[i];
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE typename MatrixDenseBase<T, Rows, Cols, D>::const_reference
MatrixDenseBase<T, Rows, Cols, D>::operator[](size_t i) const {
    assert(i < rows() * cols());
    return derived()[i];
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE D &MatrixDenseBase<T, Rows, Cols, D>::derived() {
    return static_cast<D &>(*this);
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE const D &MatrixDenseBase<T, Rows, Cols, D>::derived() const {
    return static_cast<const D &>(*this);
}

#pragma endregion

}// namespace vox
