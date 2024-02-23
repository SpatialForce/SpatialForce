//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "functors.h"
#include "core/define.h"

#include <tuple>

namespace vox {

static constexpr size_t kMatrixSizeDynamic = 0;

template<size_t Rows, size_t Cols>
constexpr bool isMatrixSizeDynamic() {
    return (Rows == kMatrixSizeDynamic) || (Cols == kMatrixSizeDynamic);
}

template<size_t Rows, size_t Cols>
constexpr bool isMatrixSizeStatic() {
    return !isMatrixSizeDynamic<Rows, Cols>();
}

template<size_t Rows, size_t Cols>
constexpr bool isMatrixStaticSquare() {
    return isMatrixSizeStatic<Rows, Cols>() && (Rows == Cols);
}

template<size_t Rows, size_t Cols>
struct IsMatrixSizeDynamic {
    static const bool value = isMatrixSizeDynamic<Rows, Cols>();
};

template<size_t Rows, size_t Cols>
struct IsMatrixSizeStatic {
    static const bool value = isMatrixSizeStatic<Rows, Cols>();
};

template<size_t Rows, size_t Cols>
struct IsMatrixSizeSquare {
    static const bool value = isMatrixStaticSquare<Rows, Cols>();
};

template<typename T, size_t Rows, size_t Cols>
class Matrix;

template<typename T, size_t Rows, size_t Cols, typename M1>
class MatrixDiagonal;

template<typename T, size_t Rows, size_t Cols, typename M1>
class MatrixOffDiagonal;

template<typename T, size_t Rows, size_t Cols, typename M1>
class MatrixTri;

template<typename T, size_t Rows, size_t Cols, typename M1>
class MatrixTranspose;

template<typename T, size_t Rows, size_t Cols, typename M1, typename UnaryOperation>
class MatrixUnaryOp;

template<typename T, size_t Rows, size_t Cols, typename M1, typename BinaryOperation>
class MatrixScalarElemWiseBinaryOp;

////////////////////////////////////////////////////////////////////////////////
// MARK: MatrixExpression

//!
//! \brief Base class for matrix expression.
//!
//! Matrix expression is a meta type that enables template expression
//! pattern.
//!
//! \tparam T  Real number type.
//! \tparam E  Subclass type.
//!
template<typename T, size_t Rows, size_t Cols, typename Derived>
class MatrixExpression {
public:
    using value_type = T;

#pragma region Core Expression Interface
    //! Returns the number of rows.
    CUDA_CALLABLE constexpr size_t rows() const;

    //! Returns the number of columns.
    CUDA_CALLABLE constexpr size_t cols() const;

    //! Returns the evaluated value for (i, j).
    CUDA_CALLABLE T eval(size_t i, size_t j) const;
#pragma endregion

#pragma region Simple getters
    CUDA_CALLABLE Matrix<T, Rows, Cols> eval() const;

    //! Returns true if this matrix is similar to the input matrix within the
    //! given tolerance.
    template<size_t R, size_t C, typename E>
    CUDA_CALLABLE bool isSimilar(const MatrixExpression<T, R, C, E> &m,
                                 double tol = thrust::numeric_limits<double>::epsilon()) const;

    //! Returns true if this matrix is a square matrix.
    CUDA_CALLABLE constexpr bool isSquare() const;

    CUDA_CALLABLE value_type sum() const;

    CUDA_CALLABLE value_type avg() const;

    CUDA_CALLABLE value_type min() const;

    CUDA_CALLABLE value_type max() const;

    CUDA_CALLABLE value_type absmin() const;

    CUDA_CALLABLE value_type absmax() const;

    CUDA_CALLABLE value_type trace() const;

    CUDA_CALLABLE value_type determinant() const;

    CUDA_CALLABLE size_t dominantAxis() const;

    CUDA_CALLABLE size_t subminantAxis() const;

    CUDA_CALLABLE value_type norm() const;

    CUDA_CALLABLE value_type normSquared() const;

    CUDA_CALLABLE value_type frobeniusNorm() const;

    CUDA_CALLABLE value_type length() const;

    CUDA_CALLABLE value_type lengthSquared() const;

    //! Returns the distance to the other vector.
    template<size_t R, size_t C, typename E>
    CUDA_CALLABLE value_type distanceTo(const MatrixExpression<T, R, C, E> &other) const;

    //! Returns the squared distance to the other vector.
    template<size_t R, size_t C, typename E>
    CUDA_CALLABLE value_type distanceSquaredTo(
        const MatrixExpression<T, R, C, E> &other) const;

    CUDA_CALLABLE MatrixScalarElemWiseBinaryOp<T, Rows, Cols, const Derived &, thrust::divides<T>>
    normalized() const;

    //! Returns diagonal part of this matrix.
    CUDA_CALLABLE MatrixDiagonal<T, Rows, Cols, const Derived &> diagonal() const;

    //! Returns off-diagonal part of this matrix.
    CUDA_CALLABLE MatrixOffDiagonal<T, Rows, Cols, const Derived &> offDiagonal() const;

    //! Returns strictly lower triangle part of this matrix.
    CUDA_CALLABLE MatrixTri<T, Rows, Cols, const Derived &> strictLowerTri() const;

    //! Returns strictly upper triangle part of this matrix.
    CUDA_CALLABLE MatrixTri<T, Rows, Cols, const Derived &> strictUpperTri() const;

    //! Returns lower triangle part of this matrix (including the diagonal).
    CUDA_CALLABLE MatrixTri<T, Rows, Cols, const Derived &> lowerTri() const;

    //! Returns upper triangle part of this matrix (including the diagonal).
    CUDA_CALLABLE MatrixTri<T, Rows, Cols, const Derived &> upperTri() const;

    CUDA_CALLABLE MatrixTranspose<T, Rows, Cols, const Derived &> transposed() const;

    //! Returns inverse matrix.
    CUDA_CALLABLE Matrix<T, Rows, Cols> inverse() const;

    template<typename U>
    CUDA_CALLABLE MatrixUnaryOp<U, Rows, Cols, const Derived &, TypeCast<T, U>> castTo() const;
#pragma endregion

#pragma region Binary Operators
    template<size_t R, size_t C, typename E, typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || Cols == 1) &&
                         (isMatrixSizeDynamic<R, C>() || C == 1),
                     U>
        CUDA_CALLABLE dot(const MatrixExpression<T, R, C, E> &expression) const;

    template<size_t R, size_t C, typename E, typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                      (Rows == 2 && Cols == 1)) &&
                         (isMatrixSizeDynamic<R, C>() || (R == 2 && C == 1)),
                     U>
        CUDA_CALLABLE cross(const MatrixExpression<T, R, C, E> &expression) const;

    template<size_t R, size_t C, typename E, typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                      (Rows == 3 && Cols == 1)) &&
                         (isMatrixSizeDynamic<R, C>() || (R == 3 && C == 1)),
                     Matrix<U, 3, 1>>
        CUDA_CALLABLE cross(const MatrixExpression<T, R, C, E> &expression) const;

    //! Returns the reflection vector to the surface with given surface normal.
    template<size_t R, size_t C, typename E, typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                      ((Rows == 2 || Rows == 3) && Cols == 1)) &&
                         (isMatrixSizeDynamic<R, C>() ||
                          ((R == 2 || R == 3) && C == 1)),
                     Matrix<U, Rows, 1>>
        CUDA_CALLABLE reflected(const MatrixExpression<T, R, C, E> &normal) const;

    //! Returns the projected vector to the surface with given surface normal.
    template<size_t R, size_t C, typename E, typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                      ((Rows == 2 || Rows == 3) && Cols == 1)) &&
                         (isMatrixSizeDynamic<R, C>() ||
                          ((R == 2 || R == 3) && C == 1)),
                     Matrix<U, Rows, 1>>
        CUDA_CALLABLE projected(const MatrixExpression<T, R, C, E> &normal) const;

    //! Returns the tangential vector for this vector.
    template<typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                      (Rows == 2 && Cols == 1)),
                     Matrix<U, 2, 1>>
        CUDA_CALLABLE tangential() const;

    //! Returns the tangential vectors for this vector.
    template<typename U = value_type>
    std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() ||
                      (Rows == 3 && Cols == 1)),
                     std::tuple<Matrix<U, 3, 1>, Matrix<U, 3, 1>>>
        CUDA_CALLABLE tangentials() const;
#pragma endregion

#pragma region Other Helpers
    //! Returns actual implementation (the subclass).
    CUDA_CALLABLE Derived &derived();

    //! Returns actual implementation (the subclass).
    CUDA_CALLABLE const Derived &derived() const;
#pragma endregion

protected:
    // Prohibits constructing this class instance.
    MatrixExpression() = default;

    CUDA_CALLABLE constexpr static T determinant(const MatrixExpression<T, 1, 1, Derived> &m);

    CUDA_CALLABLE constexpr static T determinant(const MatrixExpression<T, 2, 2, Derived> &m);

    CUDA_CALLABLE constexpr static T determinant(const MatrixExpression<T, 3, 3, Derived> &m);

    CUDA_CALLABLE constexpr static T determinant(const MatrixExpression<T, 4, 4, Derived> &m);

    template<typename U = value_type>
    static std::enable_if_t<
        (Rows > 4 && Cols > 4) || isMatrixSizeDynamic<Rows, Cols>(), U>
        CUDA_CALLABLE determinant(const MatrixExpression<T, Rows, Cols, Derived> &m);

    CUDA_CALLABLE static void inverse(const MatrixExpression<T, 1, 1, Derived> &m,
                                      Matrix<T, Rows, Cols> &result);

    CUDA_CALLABLE static void inverse(const MatrixExpression<T, 2, 2, Derived> &m,
                                      Matrix<T, Rows, Cols> &result);

    CUDA_CALLABLE static void inverse(const MatrixExpression<T, 3, 3, Derived> &m,
                                      Matrix<T, Rows, Cols> &result);

    CUDA_CALLABLE static void inverse(const MatrixExpression<T, 4, 4, Derived> &m,
                                      Matrix<T, Rows, Cols> &result);

    template<typename M = Matrix<T, Rows, Cols>>
    CUDA_CALLABLE static void inverse(const MatrixExpression &m,
                                      std::enable_if_t<(Rows > 4 && Cols > 4) ||
                                                           isMatrixSizeDynamic<Rows, Cols>(),
                                                       M> &result);
};

//====================================================================================================================
#pragma region MatrixConstant

template<typename T, size_t Rows, size_t Cols>
class MatrixConstant
    : public MatrixExpression<T, Rows, Cols, MatrixConstant<T, Rows, Cols>> {
public:
    CUDA_CALLABLE constexpr MatrixConstant(size_t r, size_t c, const T &val)
        : _rows(r), _cols(c), _val(val) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE constexpr T operator()(size_t, size_t) const;

private:
    size_t _rows;
    size_t _cols;
    T _val;
};
#pragma endregion

#pragma region MatrixDiagonal
template<typename T, size_t Rows, size_t Cols, typename M1>
class MatrixDiagonal
    : public MatrixExpression<T, Rows, Cols,
                              MatrixDiagonal<T, Rows, Cols, M1>> {
public:
    CUDA_CALLABLE constexpr MatrixDiagonal(const M1 &m1) : _m1(m1) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE T operator()(size_t i, size_t j) const;

private:
    M1 _m1;
};
#pragma endregion

#pragma region MatrixOffDiagonal
template<typename T, size_t Rows, size_t Cols, typename M1>
class MatrixOffDiagonal : public MatrixExpression<T, Rows, Cols,
                                                  MatrixOffDiagonal<T, Rows, Cols, M1>> {
public:
    CUDA_CALLABLE constexpr MatrixOffDiagonal(const M1 &m1) : _m1(m1) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE T operator()(size_t i, size_t j) const;

private:
    M1 _m1;
};
#pragma endregion

#pragma region MatrixTri
template<typename T, size_t Rows, size_t Cols, typename M1>
class MatrixTri
    : public MatrixExpression<T, Rows, Cols, MatrixTri<T, Rows, Cols, M1>> {
public:
    CUDA_CALLABLE constexpr MatrixTri(const M1 &m1, bool isUpper, bool isStrict)
        : _m1(m1), _isUpper(isUpper), _isStrict(isStrict) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE T operator()(size_t i, size_t j) const;

private:
    M1 _m1;
    bool _isUpper;
    bool _isStrict;
};
#pragma endregion

#pragma region MatrixTranspose
template<typename T, size_t Rows, size_t Cols, typename M1>
class MatrixTranspose
    : public MatrixExpression<T, Rows, Cols,
                              MatrixTranspose<T, Rows, Cols, M1>> {
public:
    CUDA_CALLABLE constexpr MatrixTranspose(const M1 &m1) : _m1(m1) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE constexpr T operator()(size_t i, size_t j) const;

private:
    M1 _m1;
};
#pragma endregion

#pragma region MARK: MatrixUnaryOp
template<typename T, size_t Rows, size_t Cols, typename M1,
         typename UnaryOperation>
class MatrixUnaryOp
    : public MatrixExpression<
          T, Rows, Cols, MatrixUnaryOp<T, Rows, Cols, M1, UnaryOperation>> {
public:
    CUDA_CALLABLE constexpr MatrixUnaryOp(const M1 &m1) : _m1(m1) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE constexpr T operator()(size_t i, size_t j) const;

private:
    M1 _m1;
    UnaryOperation _op;
};
#pragma endregion

template<typename T, size_t Rows, size_t Cols, typename M1>
using MatrixNegate = MatrixUnaryOp<T, Rows, Cols, M1, std::negate<T>>;

template<typename T, size_t Rows, size_t Cols, typename M1>
using MatrixCeil = MatrixUnaryOp<T, Rows, Cols, M1, Ceil<T>>;

template<typename T, size_t Rows, size_t Cols, typename M1>
using MatrixFloor = MatrixUnaryOp<T, Rows, Cols, M1, Floor<T>>;

template<typename T, size_t Rows, size_t Cols, typename U, typename M1>
using MatrixTypeCast = MatrixUnaryOp<U, Rows, Cols, M1, TypeCast<T, U>>;

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto ceil(const MatrixExpression<T, Rows, Cols, M1> &a);

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto floor(const MatrixExpression<T, Rows, Cols, M1> &a);

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator-(const MatrixExpression<T, Rows, Cols, M1> &m);

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixElemWiseBinaryOp
//!
//! \brief Matrix expression for element-wise binary operation.
//!
//! This matrix expression represents a binary matrix operation that takes
//! two input matrix expressions.
//!
//! \tparam T                   Real number type.
//! \tparam E1                  First input expression type.
//! \tparam E2                  Second input expression type.
//! \tparam BinaryOperation     Binary operation.
//!
template<typename T, size_t Rows, size_t Cols, typename E1, typename E2,
         typename BinaryOperation>
class MatrixElemWiseBinaryOp
    : public MatrixExpression<
          T, Rows, Cols,
          MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, BinaryOperation>> {
public:
    CUDA_CALLABLE constexpr MatrixElemWiseBinaryOp(const E1 &m1, const E2 &m2)
        : _m1(m1), _m2(m2) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE constexpr T operator()(size_t i, size_t j) const;

private:
    E1 _m1;
    E2 _m2;
    BinaryOperation _op;
};

//! Matrix expression for element-wise matrix-matrix addition.
template<typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseAdd =
    MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, thrust::plus<T>>;

//! Matrix expression for element-wise matrix-matrix subtraction.
template<typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseSub =
    MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, thrust::minus<T>>;

//! Matrix expression for element-wise matrix-matrix multiplication.
template<typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseMul =
    MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, thrust::multiplies<T>>;

//! Matrix expression for element-wise matrix-matrix division.
template<typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseDiv =
    MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, thrust::divides<T>>;

//! Matrix expression for element-wise matrix-matrix min operation.
template<typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseMin = MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, Min<T>>;

//! Matrix expression for element-wise matrix-matrix max operation.
template<typename T, size_t Rows, size_t Cols, typename E1, typename E2>
using MatrixElemWiseMax = MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, Max<T>>;

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto operator+(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b);

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto operator-(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b);

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto elemMul(const MatrixExpression<T, Rows, Cols, M1> &a,
                                     const MatrixExpression<T, Rows, Cols, M2> &b);

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto elemDiv(const MatrixExpression<T, Rows, Cols, M1> &a,
                                     const MatrixExpression<T, Rows, Cols, M2> &b);

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto min(const MatrixExpression<T, Rows, Cols, M1> &a,
                                 const MatrixExpression<T, Rows, Cols, M2> &b);

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto max(const MatrixExpression<T, Rows, Cols, M1> &a,
                                 const MatrixExpression<T, Rows, Cols, M2> &b);
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixScalarElemWiseBinaryOp
template<typename T, size_t Rows, size_t Cols, typename M1,
         typename BinaryOperation>
class MatrixScalarElemWiseBinaryOp
    : public MatrixExpression<
          T, Rows, Cols,
          MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, BinaryOperation>> {
public:
    CUDA_CALLABLE constexpr MatrixScalarElemWiseBinaryOp(const M1 &m1, const T &s2)
        : _m1(m1), _s2(s2) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE constexpr T operator()(size_t i, size_t j) const;

private:
    M1 _m1;
    T _s2;
    BinaryOperation _op;
};

template<typename T, size_t Rows, size_t Cols, typename M1>
using MatrixScalarElemWiseAdd =
    MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, thrust::plus<T>>;

template<typename T, size_t Rows, size_t Cols, typename M1>
using MatrixScalarElemWiseSub =
    MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, thrust::minus<T>>;

template<typename T, size_t Rows, size_t Cols, typename M1>
using MatrixScalarElemWiseMul =
    MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, thrust::multiplies<T>>;

template<typename T, size_t Rows, size_t Cols, typename M1>
using MatrixScalarElemWiseDiv =
    MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, thrust::divides<T>>;

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator+(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const T &b);

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator-(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const T &b);

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator*(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const T &b);

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator/(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const T &b);
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region ScalarMatrixElemWiseBinaryOp
template<typename T, size_t Rows, size_t Cols, typename M2,
         typename BinaryOperation>
class ScalarMatrixElemWiseBinaryOp
    : public MatrixExpression<
          T, Rows, Cols,
          ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, BinaryOperation>> {
public:
    CUDA_CALLABLE constexpr ScalarMatrixElemWiseBinaryOp(const T &s1, const M2 &m2)
        : _s1(s1), _m2(m2) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE constexpr T operator()(size_t i, size_t j) const;

private:
    T _s1;
    M2 _m2;
    BinaryOperation _op;
};

template<typename T, size_t Rows, size_t Cols, typename M2>
using ScalarMatrixElemWiseAdd =
    ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, thrust::plus<T>>;

template<typename T, size_t Rows, size_t Cols, typename M2>
using ScalarMatrixElemWiseSub =
    ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, thrust::minus<T>>;

template<typename T, size_t Rows, size_t Cols, typename M2>
using ScalarMatrixElemWiseMul =
    ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, thrust::multiplies<T>>;

template<typename T, size_t Rows, size_t Cols, typename M2>
using ScalarMatrixElemWiseDiv =
    ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, thrust::divides<T>>;

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M2>
CUDA_CALLABLE constexpr auto operator+(const T &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b);

template<typename T, size_t Rows, size_t Cols, typename M2>
CUDA_CALLABLE constexpr auto operator-(const T &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b);

template<typename T, size_t Rows, size_t Cols, typename M2>
CUDA_CALLABLE constexpr auto operator*(const T &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b);

template<typename T, size_t Rows, size_t Cols, typename M2>
CUDA_CALLABLE constexpr auto operator/(const T &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b);
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixTernaryOp
template<typename T, size_t Rows, size_t Cols, typename M1, typename M2,
         typename M3, typename TernaryOperation>
class MatrixTernaryOp
    : public MatrixExpression<
          T, Rows, Cols,
          MatrixTernaryOp<T, Rows, Cols, M1, M2, M3, TernaryOperation>> {
public:
    CUDA_CALLABLE constexpr MatrixTernaryOp(const M1 &m1, const M2 &m2, const M3 &m3)
        : _m1(m1), _m2(m2), _m3(m3) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE constexpr T operator()(size_t i, size_t j) const;

private:
    M1 _m1;
    M2 _m2;
    M3 _m3;
    TernaryOperation _op;
};

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2,
         typename M3>
using MatrixClamp = MatrixTernaryOp<T, Rows, Cols, M1, M2, M3, Clamp<T>>;

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2,
         typename M3>
CUDA_CALLABLE auto clamp(const MatrixExpression<T, Rows, Cols, M1> &a,
                         const MatrixExpression<T, Rows, Cols, M2> &low,
                         const MatrixExpression<T, Rows, Cols, M3> &high);
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixMul
template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
class MatrixMul
    : public MatrixExpression<T, Rows, Cols, MatrixMul<T, Rows, Cols, M1, M2>> {
public:
    CUDA_CALLABLE constexpr MatrixMul(const M1 &m1, const M2 &m2) : _m1(m1), _m2(m2) {}

    CUDA_CALLABLE constexpr size_t rows() const;

    CUDA_CALLABLE constexpr size_t cols() const;

    CUDA_CALLABLE T operator()(size_t i, size_t j) const;

private:
    M1 _m1;
    M2 _m2;
};

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M1,
         typename M2>
CUDA_CALLABLE auto operator*(const MatrixExpression<T, R1, C1, M1> &a,
                             const MatrixExpression<T, R2, C2, M2> &b);

#pragma endregion
}// namespace vox

#include "matrix_expression-inl.h"