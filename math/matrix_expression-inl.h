//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "matrix.h"

namespace vox {

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixExpression

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr size_t MatrixExpression<T, Rows, Cols, D>::rows() const {
    return static_cast<const D &>(*this).rows();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr size_t MatrixExpression<T, Rows, Cols, D>::cols() const {
    return static_cast<const D &>(*this).cols();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::eval(size_t i, size_t j) const {
    return derived()(i, j);
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE Matrix<T, Rows, Cols> MatrixExpression<T, Rows, Cols, D>::eval() const {
    return Matrix<T, Rows, Cols>(*this);
}

#pragma endregion

#pragma region Simple Getters
template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE bool MatrixExpression<T, Rows, Cols, D>::isSimilar(
    const MatrixExpression<T, R, C, E> &expression, double tol) const {
    if (expression.rows() != rows() || expression.cols() != cols()) {
        return false;
    }

    SimilarTo<T> op{tol};
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            if (!op(eval(i, j), expression.eval(i, j))) {
                return false;
            }
        }
    }
    return true;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr bool MatrixExpression<T, Rows, Cols, D>::isSquare() const {
    return rows() == cols();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::sum() const {
    T s = 0;
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            s += eval(i, j);
        }
    }
    return s;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::avg() const {
    return sum() / (rows() * cols());
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::min() const {
    T s = eval(0, 0);
    for (size_t j = 1; j < cols(); ++j) {
        s = ::min(s, eval(0, j));
    }
    for (size_t i = 1; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            s = ::min(s, eval(i, j));
        }
    }
    return s;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::max() const {
    T s = eval(0, 0);
    for (size_t j = 1; j < cols(); ++j) {
        s = ::max(s, eval(0, j));
    }
    for (size_t i = 1; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            s = ::max(s, eval(i, j));
        }
    }
    return s;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::absmin() const {
    T s = eval(0, 0);
    for (size_t j = 1; j < cols(); ++j) {
        s = vox::absmin(s, eval(0, j));
    }
    for (size_t i = 1; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            s = vox::absmin(s, eval(i, j));
        }
    }
    return s;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::absmax() const {
    T s = eval(0, 0);
    for (size_t j = 1; j < cols(); ++j) {
        s = vox::absmax(s, eval(0, j));
    }
    for (size_t i = 1; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            s = vox::absmax(s, eval(i, j));
        }
    }
    return s;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::trace() const {
    assert(rows() == cols());

    T result = eval(0, 0);
    for (size_t i = 1; i < rows(); ++i) {
        result += eval(i, i);
    }
    return result;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::determinant() const {
    assert(rows() == cols());

    return determinant(*this);
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE size_t MatrixExpression<T, Rows, Cols, D>::dominantAxis() const {
    assert(cols() == 1);

    size_t ret = 0;
    T best = eval(0, 0);
    for (size_t i = 1; i < rows(); ++i) {
        T curr = eval(i, 0);
        if (::fabs(curr) > ::fabs(best)) {
            best = curr;
            ret = i;
        }
    }
    return ret;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE size_t MatrixExpression<T, Rows, Cols, D>::subminantAxis() const {
    assert(cols() == 1);

    size_t ret = 0;
    T best = eval(0, 0);
    for (size_t i = 1; i < rows(); ++i) {
        T curr = eval(i, 0);
        if (::fabs(curr) < ::fabs(best)) {
            best = curr;
            ret = i;
        }
    }
    return ret;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::norm() const {
    return std::sqrt(normSquared());
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::normSquared() const {
    T result = 0;
    for (size_t i = 0; i < rows(); ++i) {
        for (size_t j = 0; j < cols(); ++j) {
            result += eval(i, j) * eval(i, j);
        }
    }
    return result;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::frobeniusNorm() const {
    return norm();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::length() const {
    assert(cols() == 1);
    return norm();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::lengthSquared() const {
    assert(cols() == 1);
    return normSquared();
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::distanceTo(
    const MatrixExpression<T, R, C, E> &other) const {
    assert(cols() == 1);
    return std::sqrt(distanceSquaredTo(other));
};

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E>
CUDA_CALLABLE T MatrixExpression<T, Rows, Cols, D>::distanceSquaredTo(
    const MatrixExpression<T, R, C, E> &other) const {
    assert(cols() == 1);
    return D(derived() - other.derived()).normSquared();
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE MatrixScalarElemWiseDiv<T, Rows, Cols, const D &>
MatrixExpression<T, Rows, Cols, D>::normalized() const {
    return MatrixScalarElemWiseDiv<T, Rows, Cols, const D &>{derived(), norm()};
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE MatrixDiagonal<T, Rows, Cols, const D &>
MatrixExpression<T, Rows, Cols, D>::diagonal() const {
    return MatrixDiagonal<T, Rows, Cols, const D &>{derived()};
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE MatrixOffDiagonal<T, Rows, Cols, const D &>
MatrixExpression<T, Rows, Cols, D>::offDiagonal() const {
    return MatrixOffDiagonal<T, Rows, Cols, const D &>{derived()};
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE MatrixTri<T, Rows, Cols, const D &>
MatrixExpression<T, Rows, Cols, D>::strictLowerTri() const {
    return MatrixTri<T, Rows, Cols, const D &>{derived(), false, true};
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE MatrixTri<T, Rows, Cols, const D &>
MatrixExpression<T, Rows, Cols, D>::strictUpperTri() const {
    return MatrixTri<T, Rows, Cols, const D &>{derived(), true, true};
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE MatrixTri<T, Rows, Cols, const D &>
MatrixExpression<T, Rows, Cols, D>::lowerTri() const {
    return MatrixTri<T, Rows, Cols, const D &>{derived(), false, false};
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE MatrixTri<T, Rows, Cols, const D &>
MatrixExpression<T, Rows, Cols, D>::upperTri() const {
    return MatrixTri<T, Rows, Cols, const D &>{derived(), true, false};
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE MatrixTranspose<T, Rows, Cols, const D &>
MatrixExpression<T, Rows, Cols, D>::transposed() const {
    return MatrixTranspose<T, Rows, Cols, const D &>{derived()};
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE Matrix<T, Rows, Cols> MatrixExpression<T, Rows, Cols, D>::inverse() const {
    Matrix<T, Rows, Cols> result;
    inverse(*this, result);
    return result;
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<typename U>
CUDA_CALLABLE MatrixTypeCast<T, Rows, Cols, U, const D &>
MatrixExpression<T, Rows, Cols, D>::castTo() const {
    return MatrixTypeCast<T, Rows, Cols, U, const D &>{derived()};
}

//

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E, typename U>
CUDA_CALLABLE std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || Cols == 1) && (isMatrixSizeDynamic<R, C>() || C == 1), U>
MatrixExpression<T, Rows, Cols, D>::dot(
    const MatrixExpression<T, R, C, E> &expression) const {
    assert(expression.rows() == rows() && expression.cols() == 1);

    T sum = eval(0, 0) * expression.eval(0, 0);
    for (size_t i = 1; i < rows(); ++i) {
        sum += eval(i, 0) * expression.eval(i, 0);
    }
    return sum;
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E, typename U>
CUDA_CALLABLE std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || (Rows == 2 && Cols == 1)) && (isMatrixSizeDynamic<R, C>() || (R == 2 && C == 1)), U>
MatrixExpression<T, Rows, Cols, D>::cross(
    const MatrixExpression<T, R, C, E> &expression) const {
    assert(rows() == 2 && cols() == 1 && expression.rows() == 2 &&
           expression.cols() == 1);

    return eval(0, 0) * expression.eval(1, 0) -
           expression.eval(0, 0) * eval(1, 0);
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E, typename U>
CUDA_CALLABLE std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || (Rows == 3 && Cols == 1)) && (isMatrixSizeDynamic<R, C>() || (R == 3 && C == 1)), Matrix<U, 3, 1>>
MatrixExpression<T, Rows, Cols, D>::cross(
    const MatrixExpression<T, R, C, E> &exp) const {
    assert(rows() == 3 && cols() == 1 && exp.rows() == 3 &&
           exp.cols() == 1);

    return Matrix<U, 3, 1>(
        eval(1, 0) * exp.eval(2, 0) - exp.eval(1, 0) * eval(2, 0),
        eval(2, 0) * exp.eval(0, 0) - exp.eval(2, 0) * eval(0, 0),
        eval(0, 0) * exp.eval(1, 0) - exp.eval(0, 0) * eval(1, 0));
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E, typename U>
CUDA_CALLABLE std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || ((Rows == 2 || Rows == 3) && Cols == 1)) && (isMatrixSizeDynamic<R, C>() || ((R == 2 || R == 3) && C == 1)), Matrix<U, Rows, 1>>
MatrixExpression<T, Rows, Cols, D>::reflected(
    const MatrixExpression<T, R, C, E> &normal) const {
    assert((rows() == 2 || rows() == 3) && cols() == 1 &&
           normal.rows() == rows() && normal.cols() == 1);

    // this - 2(this.n)n
    return (*this) - 2 * dot(normal) * normal;
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<size_t R, size_t C, typename E, typename U>
CUDA_CALLABLE std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || ((Rows == 2 || Rows == 3) && Cols == 1)) && (isMatrixSizeDynamic<R, C>() || ((R == 2 || R == 3) && C == 1)), Matrix<U, Rows, 1>>
MatrixExpression<T, Rows, Cols, D>::projected(
    const MatrixExpression<T, R, C, E> &normal) const {
    assert((rows() == 2 || rows() == 3) && cols() == 1 &&
           normal.rows() == rows() && normal.cols() == 1);

    // this - this.n n
    return (*this) - this->dot(normal) * normal;
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<typename U>
CUDA_CALLABLE std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || (Rows == 2 && Cols == 1)), Matrix<U, 2, 1>>
MatrixExpression<T, Rows, Cols, D>::tangential() const {
    assert(rows() == 2 && cols() == 1);

    return Matrix<U, 2, 1>{-eval(1, 0), eval(0, 0)};
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<typename U>
CUDA_CALLABLE std::enable_if_t<(isMatrixSizeDynamic<Rows, Cols>() || (Rows == 3 && Cols == 1)), thrust::tuple<Matrix<U, 3, 1>, Matrix<U, 3, 1>>>
MatrixExpression<T, Rows, Cols, D>::tangentials() const {
    assert(rows() == 3 && cols() == 1);

    using V = Matrix<T, 3, 1>;
    V a =
        ((std::fabs(eval(1, 0)) > 0 || std::fabs(eval(2, 0)) > 0) ? V(1, 0, 0) : V(0, 1, 0))
            .cross(*this)
            .normalized();
    V b = this->cross(a);
    return thrust::make_tuple(a, b);
}

//

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE D &MatrixExpression<T, Rows, Cols, D>::derived() {
    return static_cast<D &>(*this);
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE const D &MatrixExpression<T, Rows, Cols, D>::derived() const {
    return static_cast<const D &>(*this);
}

//

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr T MatrixExpression<T, Rows, Cols, D>::determinant(
    const MatrixExpression<T, 1, 1, D> &m) {
    return m.eval(0, 0);
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr T MatrixExpression<T, Rows, Cols, D>::determinant(
    const MatrixExpression<T, 2, 2, D> &m) {
    return m.eval(0, 0) * m.eval(1, 1) - m.eval(1, 0) * m.eval(0, 1);
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr T MatrixExpression<T, Rows, Cols, D>::determinant(
    const MatrixExpression<T, 3, 3, D> &m) {
    return m.eval(0, 0) * m.eval(1, 1) * m.eval(2, 2) -
           m.eval(0, 0) * m.eval(1, 2) * m.eval(2, 1) +
           m.eval(0, 1) * m.eval(1, 2) * m.eval(2, 0) -
           m.eval(0, 1) * m.eval(1, 0) * m.eval(2, 2) +
           m.eval(0, 2) * m.eval(1, 0) * m.eval(2, 1) -
           m.eval(0, 2) * m.eval(1, 1) * m.eval(2, 0);
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE constexpr T MatrixExpression<T, Rows, Cols, D>::determinant(
    const MatrixExpression<T, 4, 4, D> &m) {
    return m.eval(0, 0) * m.eval(1, 1) * m.eval(2, 2) * m.eval(3, 3) +
           m.eval(0, 0) * m.eval(1, 2) * m.eval(2, 3) * m.eval(3, 1) +
           m.eval(0, 0) * m.eval(1, 3) * m.eval(2, 1) * m.eval(3, 2) +
           m.eval(0, 1) * m.eval(1, 0) * m.eval(2, 3) * m.eval(3, 2) +
           m.eval(0, 1) * m.eval(1, 2) * m.eval(2, 0) * m.eval(3, 3) +
           m.eval(0, 1) * m.eval(1, 3) * m.eval(2, 2) * m.eval(3, 0) +
           m.eval(0, 2) * m.eval(1, 0) * m.eval(2, 1) * m.eval(3, 3) +
           m.eval(0, 2) * m.eval(1, 1) * m.eval(2, 3) * m.eval(3, 0) +
           m.eval(0, 2) * m.eval(1, 3) * m.eval(2, 0) * m.eval(3, 1) +
           m.eval(0, 3) * m.eval(1, 0) * m.eval(2, 2) * m.eval(3, 1) +
           m.eval(0, 3) * m.eval(1, 1) * m.eval(2, 0) * m.eval(3, 2) +
           m.eval(0, 3) * m.eval(1, 2) * m.eval(2, 1) * m.eval(3, 0) -
           m.eval(0, 0) * m.eval(1, 1) * m.eval(2, 3) * m.eval(3, 2) -
           m.eval(0, 0) * m.eval(1, 2) * m.eval(2, 1) * m.eval(3, 3) -
           m.eval(0, 0) * m.eval(1, 3) * m.eval(2, 2) * m.eval(3, 1) -
           m.eval(0, 1) * m.eval(1, 0) * m.eval(2, 2) * m.eval(3, 3) -
           m.eval(0, 1) * m.eval(1, 2) * m.eval(2, 3) * m.eval(3, 0) -
           m.eval(0, 1) * m.eval(1, 3) * m.eval(2, 0) * m.eval(3, 2) -
           m.eval(0, 2) * m.eval(1, 0) * m.eval(2, 3) * m.eval(3, 1) -
           m.eval(0, 2) * m.eval(1, 1) * m.eval(2, 0) * m.eval(3, 3) -
           m.eval(0, 2) * m.eval(1, 3) * m.eval(2, 1) * m.eval(3, 0) -
           m.eval(0, 3) * m.eval(1, 0) * m.eval(2, 1) * m.eval(3, 2) -
           m.eval(0, 3) * m.eval(1, 1) * m.eval(2, 2) * m.eval(3, 0) -
           m.eval(0, 3) * m.eval(1, 2) * m.eval(2, 0) * m.eval(3, 1);
}

template<typename T, size_t Rows, size_t Cols, typename D>
template<typename U>
CUDA_CALLABLE std::enable_if_t<(Rows > 4 && Cols > 4) || isMatrixSizeDynamic<Rows, Cols>(), U>
MatrixExpression<T, Rows, Cols, D>::determinant(const MatrixExpression &m) {
    // Computes inverse matrix using Gaussian elimination method.
    // https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
    Matrix<T, Rows, Cols> a{m.derived()};

    T result = 1;
    for (size_t i = 0; i < m.rows(); ++i) {
        // Search for maximum in this column
        T maxEl = ::fabs(a(i, i));
        size_t maxRow = i;
        for (size_t k = i + 1; k < m.rows(); ++k) {
            if (::fabs(a(k, i)) > maxEl) {
                maxEl = ::fabs(a(k, i));
                maxRow = k;
            }
        }

        // Swap maximum row with current row (column by column)
        if (maxRow != i) {
            for (size_t k = i; k < m.rows(); ++k) {
                thrust::swap(a(maxRow, k), a(i, k));
            }
            result *= -1;
        }

        // Make all rows below this one 0 in current column
        for (size_t k = i + 1; k < m.rows(); ++k) {
            T c = -a(k, i) / a(i, i);
            for (size_t j = i; j < m.rows(); ++j) {
                if (i == j) {
                    a(k, j) = 0;
                } else {
                    a(k, j) += c * a(i, j);
                }
            }
        }
    }

    for (size_t i = 0; i < m.rows(); ++i) {
        result *= a(i, i);
    }
    return result;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE void MatrixExpression<T, Rows, Cols, D>::inverse(
    const MatrixExpression<T, 1, 1, D> &m, Matrix<T, Rows, Cols> &result) {
    result(0, 0) = 1 / m(0, 0);
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE void MatrixExpression<T, Rows, Cols, D>::inverse(
    const MatrixExpression<T, 2, 2, D> &m, Matrix<T, Rows, Cols> &result) {
    T d = determinant(m);
    result(0, 0) = m.eval(1, 1) / d;
    result(0, 1) = -m.eval(0, 1) / d;
    result(1, 0) = -m.eval(1, 0) / d;
    result(1, 1) = m.eval(0, 0) / d;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE void MatrixExpression<T, Rows, Cols, D>::inverse(
    const MatrixExpression<T, 3, 3, D> &m, Matrix<T, Rows, Cols> &result) {
    T d = determinant(m);

    result(0, 0) =
        (m.eval(1, 1) * m.eval(2, 2) - m.eval(1, 2) * m.eval(2, 1)) / d;
    result(0, 1) =
        (m.eval(0, 2) * m.eval(2, 1) - m.eval(0, 1) * m.eval(2, 2)) / d;
    result(0, 2) =
        (m.eval(0, 1) * m.eval(1, 2) - m.eval(0, 2) * m.eval(1, 1)) / d;
    result(1, 0) =
        (m.eval(1, 2) * m.eval(2, 0) - m.eval(1, 0) * m.eval(2, 2)) / d;
    result(1, 1) =
        (m.eval(0, 0) * m.eval(2, 2) - m.eval(0, 2) * m.eval(2, 0)) / d;
    result(1, 2) =
        (m.eval(0, 2) * m.eval(1, 0) - m.eval(0, 0) * m.eval(1, 2)) / d;
    result(2, 0) =
        (m.eval(1, 0) * m.eval(2, 1) - m.eval(1, 1) * m.eval(2, 0)) / d;
    result(2, 1) =
        (m.eval(0, 1) * m.eval(2, 0) - m.eval(0, 0) * m.eval(2, 1)) / d;
    result(2, 2) =
        (m.eval(0, 0) * m.eval(1, 1) - m.eval(0, 1) * m.eval(1, 0)) / d;
}

template<typename T, size_t Rows, size_t Cols, typename D>
CUDA_CALLABLE void MatrixExpression<T, Rows, Cols, D>::inverse(
    const MatrixExpression<T, 4, 4, D> &m, Matrix<T, Rows, Cols> &result) {
    T d = determinant(m);
    result(0, 0) = (m.eval(1, 1) * m.eval(2, 2) * m.eval(3, 3) +
                    m.eval(1, 2) * m.eval(2, 3) * m.eval(3, 1) +
                    m.eval(1, 3) * m.eval(2, 1) * m.eval(3, 2) -
                    m.eval(1, 1) * m.eval(2, 3) * m.eval(3, 2) -
                    m.eval(1, 2) * m.eval(2, 1) * m.eval(3, 3) -
                    m.eval(1, 3) * m.eval(2, 2) * m.eval(3, 1)) /
                   d;
    result(0, 1) = (m.eval(0, 1) * m.eval(2, 3) * m.eval(3, 2) +
                    m.eval(0, 2) * m.eval(2, 1) * m.eval(3, 3) +
                    m.eval(0, 3) * m.eval(2, 2) * m.eval(3, 1) -
                    m.eval(0, 1) * m.eval(2, 2) * m.eval(3, 3) -
                    m.eval(0, 2) * m.eval(2, 3) * m.eval(3, 1) -
                    m.eval(0, 3) * m.eval(2, 1) * m.eval(3, 2)) /
                   d;
    result(0, 2) = (m.eval(0, 1) * m.eval(1, 2) * m.eval(3, 3) +
                    m.eval(0, 2) * m.eval(1, 3) * m.eval(3, 1) +
                    m.eval(0, 3) * m.eval(1, 1) * m.eval(3, 2) -
                    m.eval(0, 1) * m.eval(1, 3) * m.eval(3, 2) -
                    m.eval(0, 2) * m.eval(1, 1) * m.eval(3, 3) -
                    m.eval(0, 3) * m.eval(1, 2) * m.eval(3, 1)) /
                   d;
    result(0, 3) = (m.eval(0, 1) * m.eval(1, 3) * m.eval(2, 2) +
                    m.eval(0, 2) * m.eval(1, 1) * m.eval(2, 3) +
                    m.eval(0, 3) * m.eval(1, 2) * m.eval(2, 1) -
                    m.eval(0, 1) * m.eval(1, 2) * m.eval(2, 3) -
                    m.eval(0, 2) * m.eval(1, 3) * m.eval(2, 1) -
                    m.eval(0, 3) * m.eval(1, 1) * m.eval(2, 2)) /
                   d;
    result(1, 0) = (m.eval(1, 0) * m.eval(2, 3) * m.eval(3, 2) +
                    m.eval(1, 2) * m.eval(2, 0) * m.eval(3, 3) +
                    m.eval(1, 3) * m.eval(2, 2) * m.eval(3, 0) -
                    m.eval(1, 0) * m.eval(2, 2) * m.eval(3, 3) -
                    m.eval(1, 2) * m.eval(2, 3) * m.eval(3, 0) -
                    m.eval(1, 3) * m.eval(2, 0) * m.eval(3, 2)) /
                   d;
    result(1, 1) = (m.eval(0, 0) * m.eval(2, 2) * m.eval(3, 3) +
                    m.eval(0, 2) * m.eval(2, 3) * m.eval(3, 0) +
                    m.eval(0, 3) * m.eval(2, 0) * m.eval(3, 2) -
                    m.eval(0, 0) * m.eval(2, 3) * m.eval(3, 2) -
                    m.eval(0, 2) * m.eval(2, 0) * m.eval(3, 3) -
                    m.eval(0, 3) * m.eval(2, 2) * m.eval(3, 0)) /
                   d;
    result(1, 2) = (m.eval(0, 0) * m.eval(1, 3) * m.eval(3, 2) +
                    m.eval(0, 2) * m.eval(1, 0) * m.eval(3, 3) +
                    m.eval(0, 3) * m.eval(1, 2) * m.eval(3, 0) -
                    m.eval(0, 0) * m.eval(1, 2) * m.eval(3, 3) -
                    m.eval(0, 2) * m.eval(1, 3) * m.eval(3, 0) -
                    m.eval(0, 3) * m.eval(1, 0) * m.eval(3, 2)) /
                   d;
    result(1, 3) = (m.eval(0, 0) * m.eval(1, 2) * m.eval(2, 3) +
                    m.eval(0, 2) * m.eval(1, 3) * m.eval(2, 0) +
                    m.eval(0, 3) * m.eval(1, 0) * m.eval(2, 2) -
                    m.eval(0, 0) * m.eval(1, 3) * m.eval(2, 2) -
                    m.eval(0, 2) * m.eval(1, 0) * m.eval(2, 3) -
                    m.eval(0, 3) * m.eval(1, 2) * m.eval(2, 0)) /
                   d;
    result(2, 0) = (m.eval(1, 0) * m.eval(2, 1) * m.eval(3, 3) +
                    m.eval(1, 1) * m.eval(2, 3) * m.eval(3, 0) +
                    m.eval(1, 3) * m.eval(2, 0) * m.eval(3, 1) -
                    m.eval(1, 0) * m.eval(2, 3) * m.eval(3, 1) -
                    m.eval(1, 1) * m.eval(2, 0) * m.eval(3, 3) -
                    m.eval(1, 3) * m.eval(2, 1) * m.eval(3, 0)) /
                   d;
    result(2, 1) = (m.eval(0, 0) * m.eval(2, 3) * m.eval(3, 1) +
                    m.eval(0, 1) * m.eval(2, 0) * m.eval(3, 3) +
                    m.eval(0, 3) * m.eval(2, 1) * m.eval(3, 0) -
                    m.eval(0, 0) * m.eval(2, 1) * m.eval(3, 3) -
                    m.eval(0, 1) * m.eval(2, 3) * m.eval(3, 0) -
                    m.eval(0, 3) * m.eval(2, 0) * m.eval(3, 1)) /
                   d;
    result(2, 2) = (m.eval(0, 0) * m.eval(1, 1) * m.eval(3, 3) +
                    m.eval(0, 1) * m.eval(1, 3) * m.eval(3, 0) +
                    m.eval(0, 3) * m.eval(1, 0) * m.eval(3, 1) -
                    m.eval(0, 0) * m.eval(1, 3) * m.eval(3, 1) -
                    m.eval(0, 1) * m.eval(1, 0) * m.eval(3, 3) -
                    m.eval(0, 3) * m.eval(1, 1) * m.eval(3, 0)) /
                   d;
    result(2, 3) = (m.eval(0, 0) * m.eval(1, 3) * m.eval(2, 1) +
                    m.eval(0, 1) * m.eval(1, 0) * m.eval(2, 3) +
                    m.eval(0, 3) * m.eval(1, 1) * m.eval(2, 0) -
                    m.eval(0, 0) * m.eval(1, 1) * m.eval(2, 3) -
                    m.eval(0, 1) * m.eval(1, 3) * m.eval(2, 0) -
                    m.eval(0, 3) * m.eval(1, 0) * m.eval(2, 1)) /
                   d;
    result(3, 0) = (m.eval(1, 0) * m.eval(2, 2) * m.eval(3, 1) +
                    m.eval(1, 1) * m.eval(2, 0) * m.eval(3, 2) +
                    m.eval(1, 2) * m.eval(2, 1) * m.eval(3, 0) -
                    m.eval(1, 0) * m.eval(2, 1) * m.eval(3, 2) -
                    m.eval(1, 1) * m.eval(2, 2) * m.eval(3, 0) -
                    m.eval(1, 2) * m.eval(2, 0) * m.eval(3, 1)) /
                   d;
    result(3, 1) = (m.eval(0, 0) * m.eval(2, 1) * m.eval(3, 2) +
                    m.eval(0, 1) * m.eval(2, 2) * m.eval(3, 0) +
                    m.eval(0, 2) * m.eval(2, 0) * m.eval(3, 1) -
                    m.eval(0, 0) * m.eval(2, 2) * m.eval(3, 1) -
                    m.eval(0, 1) * m.eval(2, 0) * m.eval(3, 2) -
                    m.eval(0, 2) * m.eval(2, 1) * m.eval(3, 0)) /
                   d;
    result(3, 2) = (m.eval(0, 0) * m.eval(1, 2) * m.eval(3, 1) +
                    m.eval(0, 1) * m.eval(1, 0) * m.eval(3, 2) +
                    m.eval(0, 2) * m.eval(1, 1) * m.eval(3, 0) -
                    m.eval(0, 0) * m.eval(1, 1) * m.eval(3, 2) -
                    m.eval(0, 1) * m.eval(1, 2) * m.eval(3, 0) -
                    m.eval(0, 2) * m.eval(1, 0) * m.eval(3, 1)) /
                   d;
    result(3, 3) = (m.eval(0, 0) * m.eval(1, 1) * m.eval(2, 2) +
                    m.eval(0, 1) * m.eval(1, 2) * m.eval(2, 0) +
                    m.eval(0, 2) * m.eval(1, 0) * m.eval(2, 1) -
                    m.eval(0, 0) * m.eval(1, 2) * m.eval(2, 1) -
                    m.eval(0, 1) * m.eval(1, 0) * m.eval(2, 2) -
                    m.eval(0, 2) * m.eval(1, 1) * m.eval(2, 0)) /
                   d;
}

template<typename T, size_t Rows, size_t Cols, typename Derived>
template<typename M>
CUDA_CALLABLE void MatrixExpression<T, Rows, Cols, Derived>::inverse(
    const MatrixExpression<T, Rows, Cols, Derived> &m,
    std::enable_if_t<(Rows > 4 && Cols > 4) || isMatrixSizeDynamic<Rows, Cols>(), M> &result) {
    // Computes inverse matrix using Gaussian elimination method.
    // https://martin-thoma.com/solving-linear-equations-with-gaussian-elimination/
    Matrix<T, Rows, Cols> a{m.derived()};

    using ConstType = MatrixConstant<T, Rows, Cols>;
    result = MatrixDiagonal<T, Rows, Cols, ConstType>{
        ConstType{a.rows(), a.cols(), 1}};
    size_t n = m.rows();

    for (size_t i = 0; i < n; ++i) {
        // Search for maximum in this column
        T maxEl = std::fabs(a(i, i));
        size_t maxRow = i;
        for (size_t k = i + 1; k < n; ++k) {
            if (std::fabs(a(k, i)) > maxEl) {
                maxEl = std::fabs(a(k, i));
                maxRow = k;
            }
        }

        // Swap maximum row with current row (column by column)
        if (maxRow != i) {
            for (size_t k = i; k < n; ++k) {
                thrust::swap(a(maxRow, k), a(i, k));
            }
            for (size_t k = 0; k < n; ++k) {
                thrust::swap(result(maxRow, k), result(i, k));
            }
        }

        // Make all rows except this one 0 in current column
        for (size_t k = 0; k < n; ++k) {
            if (k == i) {
                continue;
            }
            T c = -a(k, i) / a(i, i);
            for (size_t j = 0; j < n; ++j) {
                result(k, j) += c * result(i, j);
                if (i == j) {
                    a(k, j) = 0;
                } else if (i < j) {
                    a(k, j) += c * a(i, j);
                }
            }
        }

        // Scale
        for (size_t k = 0; k < n; ++k) {
            T c = 1 / a(k, k);
            for (size_t j = 0; j < n; ++j) {
                a(k, j) *= c;
                result(k, j) *= c;
            }
        }
    }
}
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixConstant
template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE constexpr size_t MatrixConstant<T, Rows, Cols>::rows() const {
    return _rows;
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE constexpr size_t MatrixConstant<T, Rows, Cols>::cols() const {
    return _cols;
}

template<typename T, size_t Rows, size_t Cols>
CUDA_CALLABLE constexpr T MatrixConstant<T, Rows, Cols>::operator()(size_t, size_t) const {
    return _val;
}
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixDiagonal

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr size_t MatrixDiagonal<T, Rows, Cols, M1>::rows() const {
    return _m1.rows();
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr size_t MatrixDiagonal<T, Rows, Cols, M1>::cols() const {
    return _m1.cols();
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE T MatrixDiagonal<T, Rows, Cols, M1>::operator()(size_t i, size_t j) const {
    if (i == j) {
        return _m1(i, j);
    } else {
        return T{};
    }
}
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixOffDiagonal
template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr size_t MatrixOffDiagonal<T, Rows, Cols, M1>::rows() const {
    return _m1.rows();
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr size_t MatrixOffDiagonal<T, Rows, Cols, M1>::cols() const {
    return _m1.cols();
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE T MatrixOffDiagonal<T, Rows, Cols, M1>::operator()(size_t i, size_t j) const {
    if (i != j) {
        return _m1(i, j);
    } else {
        return T{};
    }
}
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixTri
template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr size_t MatrixTri<T, Rows, Cols, M1>::rows() const {
    return _m1.rows();
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr size_t MatrixTri<T, Rows, Cols, M1>::cols() const {
    return _m1.cols();
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE T MatrixTri<T, Rows, Cols, M1>::operator()(size_t i, size_t j) const {
    if (_isUpper) {
        if (_isStrict) {
            return (j > i) ? _m1(i, j) : 0;
        } else {
            return (j >= i) ? _m1(i, j) : 0;
        }
    } else {
        if (_isStrict) {
            return (j < i) ? _m1(i, j) : 0;
        } else {
            return (j <= i) ? _m1(i, j) : 0;
        }
    }
}
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixTranspose
template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr size_t MatrixTranspose<T, Rows, Cols, M1>::rows() const {
    return _m1.cols();
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr size_t MatrixTranspose<T, Rows, Cols, M1>::cols() const {
    return _m1.rows();
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr T MatrixTranspose<T, Rows, Cols, M1>::operator()(size_t i,
                                                                         size_t j) const {
    return _m1(j, i);
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixUnaryOp

template<typename T, size_t Rows, size_t Cols, typename M1, typename UOp>
CUDA_CALLABLE constexpr size_t MatrixUnaryOp<T, Rows, Cols, M1, UOp>::rows() const {
    return _m1.rows();
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename UOp>
CUDA_CALLABLE constexpr size_t MatrixUnaryOp<T, Rows, Cols, M1, UOp>::cols() const {
    return _m1.cols();
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename UOp>
CUDA_CALLABLE constexpr T MatrixUnaryOp<T, Rows, Cols, M1, UOp>::operator()(size_t i,
                                                                            size_t j) const {
    return _op(_m1(i, j));
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto ceil(const MatrixExpression<T, Rows, Cols, M1> &a) {
    return MatrixCeil<T, Rows, Cols, const M1 &>{a.derived()};
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto floor(const MatrixExpression<T, Rows, Cols, M1> &a) {
    return MatrixFloor<T, Rows, Cols, const M1 &>{a.derived()};
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator-(const MatrixExpression<T, Rows, Cols, M1> &m) {
    return MatrixNegate<T, Rows, Cols, const M1 &>{m.derived()};
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixElemWiseBinaryOp

template<typename T, size_t Rows, size_t Cols, typename E1, typename E2,
         typename BOp>
CUDA_CALLABLE constexpr size_t MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, BOp>::rows() const {
    return _m1.rows();
}

template<typename T, size_t Rows, size_t Cols, typename E1, typename E2,
         typename BOp>
CUDA_CALLABLE constexpr size_t MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, BOp>::cols() const {
    return _m1.cols();
}

template<typename T, size_t Rows, size_t Cols, typename E1, typename E2,
         typename BOp>
CUDA_CALLABLE constexpr T MatrixElemWiseBinaryOp<T, Rows, Cols, E1, E2, BOp>::operator()(
    size_t i, size_t j) const {
    return _op(_m1(i, j), _m2(i, j));
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto operator+(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b) {
    return MatrixElemWiseAdd<T, Rows, Cols, const M1 &, const M2 &>{a.derived(), b.derived()};
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto operator-(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b) {
    return MatrixElemWiseSub<T, Rows, Cols, const M1 &, const M2 &>{a.derived(), b.derived()};
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto elemMul(const MatrixExpression<T, Rows, Cols, M1> &a,
                                     const MatrixExpression<T, Rows, Cols, M2> &b) {
    return MatrixElemWiseMul<T, Rows, Cols, const M1 &, const M2 &>{a.derived(), b.derived()};
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto elemDiv(const MatrixExpression<T, Rows, Cols, M1> &a,
                                     const MatrixExpression<T, Rows, Cols, M2> &b) {
    return MatrixElemWiseDiv<T, Rows, Cols, const M1 &, const M2 &>{a.derived(), b.derived()};
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto min(const MatrixExpression<T, Rows, Cols, M1> &a,
                                 const MatrixExpression<T, Rows, Cols, M2> &b) {
    return MatrixElemWiseMin<T, Rows, Cols, const M1 &, const M2 &>{a.derived(), b.derived()};
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr auto max(const MatrixExpression<T, Rows, Cols, M1> &a,
                                 const MatrixExpression<T, Rows, Cols, M2> &b) {
    return MatrixElemWiseMax<T, Rows, Cols, const M1 &, const M2 &>{a.derived(), b.derived()};
}
#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixScalarElemWiseBinaryOp

template<typename T, size_t Rows, size_t Cols, typename M1, typename BOp>
CUDA_CALLABLE constexpr size_t MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, BOp>::rows() const {
    return _m1.rows();
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename BOp>
CUDA_CALLABLE constexpr size_t MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, BOp>::cols() const {
    return _m1.cols();
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename BOp>
CUDA_CALLABLE constexpr T MatrixScalarElemWiseBinaryOp<T, Rows, Cols, M1, BOp>::operator()(
    size_t i, size_t j) const {
    return _op(_m1(i, j), _s2);
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator+(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const T &b) {
    return MatrixScalarElemWiseAdd<T, Rows, Cols, const M1 &>{a.derived(), b};
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator-(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const T &b) {
    return MatrixScalarElemWiseSub<T, Rows, Cols, const M1 &>{a.derived(), b};
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator*(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const T &b) {
    return MatrixScalarElemWiseMul<T, Rows, Cols, const M1 &>{a.derived(), b};
}

template<typename T, size_t Rows, size_t Cols, typename M1>
CUDA_CALLABLE constexpr auto operator/(const MatrixExpression<T, Rows, Cols, M1> &a,
                                       const T &b) {
    return MatrixScalarElemWiseDiv<T, Rows, Cols, const M1 &>{a.derived(), b};
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region ScalarMatrixElemWiseBinaryOp

template<typename T, size_t Rows, size_t Cols, typename M2, typename BOp>
CUDA_CALLABLE constexpr size_t ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, BOp>::rows()
    const {
    return _m2.rows();
}

template<typename T, size_t Rows, size_t Cols, typename M2, typename BOp>
CUDA_CALLABLE constexpr size_t ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, BOp>::cols()
    const {
    return _m2.cols();
}

template<typename T, size_t Rows, size_t Cols, typename M2, typename BOp>
CUDA_CALLABLE constexpr T ScalarMatrixElemWiseBinaryOp<T, Rows, Cols, M2, BOp>::operator()(
    size_t i, size_t j) const {
    return _op(_s1, _m2(i, j));
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M2>
CUDA_CALLABLE constexpr auto operator+(const T &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b) {
    return ScalarMatrixElemWiseAdd<T, Rows, Cols, const M2 &>{a, b.derived()};
}

template<typename T, size_t Rows, size_t Cols, typename M2>
CUDA_CALLABLE constexpr auto operator-(const T &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b) {
    return ScalarMatrixElemWiseSub<T, Rows, Cols, const M2 &>{a, b.derived()};
}

template<typename T, size_t Rows, size_t Cols, typename M2>
CUDA_CALLABLE constexpr auto operator*(const T &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b) {
    return ScalarMatrixElemWiseMul<T, Rows, Cols, const M2 &>{a, b.derived()};
}

template<typename T, size_t Rows, size_t Cols, typename M2>
CUDA_CALLABLE constexpr auto operator/(const T &a,
                                       const MatrixExpression<T, Rows, Cols, M2> &b) {
    return ScalarMatrixElemWiseDiv<T, Rows, Cols, const M2 &>{a, b.derived()};
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixTernaryOp

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2,
         typename M3, typename TOp>
CUDA_CALLABLE constexpr size_t MatrixTernaryOp<T, Rows, Cols, M1, M2, M3, TOp>::rows() const {
    return _m1.rows();
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2,
         typename M3, typename TOp>
CUDA_CALLABLE constexpr size_t MatrixTernaryOp<T, Rows, Cols, M1, M2, M3, TOp>::cols() const {
    return _m1.cols();
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2,
         typename M3, typename TOp>
CUDA_CALLABLE constexpr T MatrixTernaryOp<T, Rows, Cols, M1, M2, M3, TOp>::operator()(
    size_t i, size_t j) const {
    return _op(_m1(i, j), _m2(i, j), _m3(i, j));
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2,
         typename M3>
CUDA_CALLABLE auto clamp(const MatrixExpression<T, Rows, Cols, M1> &a,
                         const MatrixExpression<T, Rows, Cols, M2> &low,
                         const MatrixExpression<T, Rows, Cols, M3> &high) {
    assert(a.rows() == low.rows() && a.rows() == high.rows());
    assert(a.cols() == low.cols() && a.cols() == high.cols());
    return MatrixClamp<T, Rows, Cols, const M1 &, const M2 &, const M3 &>{
        a.derived(), low.derived(), high.derived()};
}

#pragma endregion

////////////////////////////////////////////////////////////////////////////////
#pragma region MatrixMul

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr size_t MatrixMul<T, Rows, Cols, M1, M2>::rows() const {
    return _m1.rows();
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE constexpr size_t MatrixMul<T, Rows, Cols, M1, M2>::cols() const {
    return _m2.cols();
}

template<typename T, size_t Rows, size_t Cols, typename M1, typename M2>
CUDA_CALLABLE T MatrixMul<T, Rows, Cols, M1, M2>::operator()(size_t i, size_t j) const {
    T sum = _m1(i, 0) * _m2(0, j);
    for (size_t k = 1; k < _m1.cols(); ++k) {
        sum += _m1(i, k) * _m2(k, j);
    }
    return sum;
}

////////////////////////////////////////////////////////////////////////////////

template<typename T, size_t R1, size_t C1, size_t R2, size_t C2, typename M1,
         typename M2>
CUDA_CALLABLE auto operator*(const MatrixExpression<T, R1, C1, M1> &a,
                             const MatrixExpression<T, R2, C2, M2> &b) {
    assert(a.cols() == b.rows());

    return MatrixMul<T, R1, C2, const M1 &, const M2 &>{a.derived(), b.derived()};
}

#pragma endregion

}// namespace vox