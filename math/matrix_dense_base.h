//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "functors.h"
#include "matrix_expression.h"
#include "nested_initializer_list.h"

namespace vox {

// Derived type should be constructible.
template<typename T, size_t Rows, size_t Cols, typename Derived>
class MatrixDenseBase {
public:
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;

    // MARK: Simple setters/modifiers

    //! Copies from generic expression.
    template<size_t R, size_t C, typename E>
    CUDA_CALLABLE void copyFrom(const MatrixExpression<T, R, C, E> &expression);

    //! Sets diagonal elements with input scalar.
    CUDA_CALLABLE void setDiagonal(const_reference val);

    //! Sets off-diagonal elements with input scalar.
    CUDA_CALLABLE void setOffDiagonal(const_reference val);

    //! Sets i-th row with input column vector.
    template<size_t R, size_t C, typename E>
    CUDA_CALLABLE void setRow(size_t i, const MatrixExpression<T, R, C, E> &row);

    //! Sets i-th column with input vector.
    template<size_t R, size_t C, typename E>
    CUDA_CALLABLE void setColumn(size_t i, const MatrixExpression<T, R, C, E> &col);

    CUDA_CALLABLE void normalize();

    //! Transposes this matrix.
    CUDA_CALLABLE void transpose();

    //! Inverts this matrix.
    CUDA_CALLABLE void invert();

    // MARK: Operator Overloadings

    CUDA_CALLABLE reference operator()(size_t i, size_t j);

    CUDA_CALLABLE const_reference operator()(size_t i, size_t j) const;

    //! Copies from generic expression
    template<size_t R, size_t C, typename E>
    CUDA_CALLABLE MatrixDenseBase &operator=(const MatrixExpression<T, R, C, E> &expression);

    // MARK: Builders

    //! Makes a static matrix with zero entries.
    template<typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), D> makeZero();

    //! Makes a dynamic matrix with zero entries.
    template<typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), D> makeZero(
        size_t rows, size_t cols);

    //! Makes a static matrix with constant entries.
    template<typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixSizeStatic<Rows, Cols>(), D> makeConstant(
        value_type val);

    //! Makes a dynamic matrix with constant entries.
    template<typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), D> makeConstant(
        size_t rows, size_t cols, value_type val);

    //! Makes a static identity matrix.
    template<typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
    makeIdentity();

    //! Makes a dynamic identity matrix.
    template<typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixSizeDynamic<Rows, Cols>(), D> makeIdentity(
        size_t rows);

    //! Makes scale matrix.
    template<typename... Args, typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
    makeScaleMatrix(value_type first, Args... rest);

    //! Makes scale matrix.
    template<size_t R, size_t C, typename E, typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>(), D>
    makeScaleMatrix(const MatrixExpression<T, R, C, E> &expression);

    //! Makes rotation matrix.
    //! \warning Input angle should be radian.
    template<typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 2),
                                          D>
    makeRotationMatrix(T rad);

    //! Makes rotation matrix.
    //! \warning Input angle should be radian.
    template<size_t R, size_t C, typename E, typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<
        isMatrixStaticSquare<Rows, Cols>() && (Rows == 3 || Rows == 4), D>
    makeRotationMatrix(const MatrixExpression<T, R, C, E> &axis, T rad);

    //! Makes translation matrix.
    template<size_t R, size_t C, typename E, typename D = Derived>
    CUDA_CALLABLE static std::enable_if_t<isMatrixStaticSquare<Rows, Cols>() && (Rows == 4),
                                          D>
    makeTranslationMatrix(const MatrixExpression<T, R, C, E> &t);

protected:
    MatrixDenseBase() = default;

private:
    // MARK: Private Helpers

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE auto begin();

    CUDA_CALLABLE constexpr auto begin() const;

    CUDA_CALLABLE auto end();

    CUDA_CALLABLE constexpr auto end() const;

    CUDA_CALLABLE reference operator[](size_t i);

    CUDA_CALLABLE const_reference operator[](size_t i) const;

    CUDA_CALLABLE Derived &derived();

    CUDA_CALLABLE const Derived &derived() const;
};

}// namespace vox

#include "matrix_dense_base-inl.h"
