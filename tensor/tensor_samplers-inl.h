//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {

namespace internal {

template<typename T, size_t N, size_t I>
struct Lerp {
    using ScalarType = typename GetScalarType<T>::value;

    template<typename View, typename... RemainingIndices>
    static auto call(const View &view, Vector<ssize_t, N> i,
                     Vector<ScalarType, N> t, RemainingIndices... indices) {
        using Next = Lerp<T, N, I - 1>;
        return lerp(Next::call(view, i, t, i[I - 1], indices...),
                    Next::call(view, i, t, i[I - 1] + 1, indices...), t[I - 1]);
    }
};

template<typename T, size_t N>
struct Lerp<T, N, 1> {
    using ScalarType = typename GetScalarType<T>::value;

    template<typename View, typename... RemainingIndices>
    static auto call(const View &view, Vector<ssize_t, N> i,
                     Vector<ScalarType, N> t, RemainingIndices... indices) {
        return lerp(view(i[0], indices...), view(i[0] + 1, indices...), t[0]);
    }
};

template<typename T, size_t N, size_t I>
struct Cubic {
    using ScalarType = typename GetScalarType<T>::value;

    template<typename View, typename CubicInterpolationOp,
             typename... RemainingIndices>
    static auto call(const View &view, Vector<ssize_t, N> i,
                     Vector<ScalarType, N> t, CubicInterpolationOp op,
                     RemainingIndices... indices) {
        using Next = Cubic<T, N, I - 1>;
        return op(
            Next::call(view, i, t, op,
                       std::max(i[I - 1] - 1, (ssize_t)view.shape()[I - 1] - 1),
                       indices...),
            Next::call(view, i, t, op, i[I - 1], indices...),
            Next::call(view, i, t, op, i[I - 1] + 1, indices...),
            Next::call(view, i, t, op,
                       std::min(i[I - 1] + 2, (ssize_t)view.shape()[I - 1] - 1),
                       indices...),
            t[I - 1]);
    }
};

template<typename T, size_t N>
struct Cubic<T, N, 1> {
    using ScalarType = typename GetScalarType<T>::value;

    template<typename View, typename CubicInterpolationOp,
             typename... RemainingIndices>
    static auto call(const View &view, Vector<ssize_t, N> i,
                     Vector<ScalarType, N> t, CubicInterpolationOp op,
                     RemainingIndices... indices) {
        return op(
            view(std::max(i[0] - 1, (ssize_t)view.shape()[0] - 1), indices...),
            view(i[0], indices...), view(i[0] + 1, indices...),
            view(std::min(i[0] + 2, (ssize_t)view.shape()[0] - 1), indices...),
            t[0]);
    }
};

template<typename T, size_t N, size_t I>
struct GetCoordinatesAndWeights {
    using ScalarType = typename GetScalarType<T>::value;

    template<typename Coords, typename Weights, typename... RemainingIndices>
    static void call(Coords &c, Weights &w, Vector<size_t, N> i,
                     Vector<ScalarType, N> t, T acc, RemainingIndices... idx) {
        using Next = GetCoordinatesAndWeights<T, N, I - 1>;
        Next::call(c, w, i, t, acc * (1 - t[I - 1]), 0, idx...);
        Next::call(c, w, i, t, acc * t[I - 1], 1, idx...);
    }
};

template<typename T, size_t N>
struct GetCoordinatesAndWeights<T, N, 1> {
    using ScalarType = typename GetScalarType<T>::value;

    template<typename Coords, typename Weights, typename... RemainingIndices>
    static void call(Coords &c, Weights &w, Vector<size_t, N> i,
                     Vector<ScalarType, N> t, T acc, RemainingIndices... idx) {
        c(0, idx...) = Vector<size_t, N>(0, idx...) + i;
        c(1, idx...) = Vector<size_t, N>(1, idx...) + i;

        w(0, idx...) = acc * (1 - t[0]);
        w(1, idx...) = acc * (t[0]);
    }
};

template<typename T, size_t N, size_t I>
struct GetCoordinatesAndGradientWeights {
    template<typename Coords, typename Weights, typename... RemainingIndices>
    static void call(Coords &c, Weights &w, Vector<size_t, N> i, Vector<T, N> t,
                     Vector<T, N> acc, RemainingIndices... idx) {
        Vector<T, N> w0 = Vector<T, N>::makeConstant(1 - t[I - 1]);
        w0[I - 1] = -1;
        Vector<T, N> w1 = Vector<T, N>::makeConstant(t[I - 1]);
        w1[I - 1] = 1;

        using Next = GetCoordinatesAndGradientWeights<T, N, I - 1>;
        Next::call(c, w, i, t, elemMul(acc, w0), 0, idx...);
        Next::call(c, w, i, t, elemMul(acc, w1), 1, idx...);
    }
};

template<typename T, size_t N>
struct GetCoordinatesAndGradientWeights<T, N, 1> {
    template<typename Coords, typename Weights, typename... RemainingIndices>
    static void call(Coords &c, Weights &w, Vector<size_t, N> i, Vector<T, N> t,
                     Vector<T, N> acc, RemainingIndices... idx) {
        c(0, idx...) = Vector<size_t, N>(0, idx...) + i;
        c(1, idx...) = Vector<size_t, N>(1, idx...) + i;

        Vector<T, N> w0 = Vector<T, N>::makeConstant(1 - t[0]);
        w0[0] = -1;
        Vector<T, N> w1 = Vector<T, N>::makeConstant(t[0]);
        w1[0] = 1;

        w(0, idx...) = elemMul(acc, w0);
        w(1, idx...) = elemMul(acc, w1);
    }
};

}// namespace internal

////////////////////////////////////////////////////////////////////////////////
// MARK: NearestTensorSampler

template<typename T, size_t N>
NearestTensorSampler<T, N>::NearestTensorSampler(
    const TensorView<const T, N> &view, const VectorType &gridSpacing,
    const VectorType &gridOrigin)
    : _view(view),
      _gridSpacing(gridSpacing),
      _invGridSpacing(ScalarType{1} / gridSpacing),
      _gridOrigin(gridOrigin) {}

template<typename T, size_t N>
NearestTensorSampler<T, N>::NearestTensorSampler(const NearestTensorSampler &other)
    : _view(other._view),
      _gridSpacing(other._gridSpacing),
      _invGridSpacing(other._invGridSpacing),
      _gridOrigin(other._gridOrigin) {}

template<typename T, size_t N>
T NearestTensorSampler<T, N>::operator()(const VectorType &pt) const {
    return _view(getCoordinate(pt));
}

template<typename T, size_t N>
typename NearestTensorSampler<T, N>::CoordIndexType
NearestTensorSampler<T, N>::getCoordinate(const VectorType &pt) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> shape = _view.shape().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, shape[i], is[i], ts[i]);
        is[i] =
            std::min(static_cast<ssize_t>(is[i] + ts[i] + 0.5), shape[i] - 1);
    }

    return is.template castTo<size_t>();
}

template<typename T, size_t N>
std::function<T(const typename NearestTensorSampler<T, N>::VectorType &)>
NearestTensorSampler<T, N>::functor() const {
    NearestTensorSampler sampler(*this);
    return [sampler](const VectorType &x) -> T { return sampler(x); };
}

////////////////////////////////////////////////////////////////////////////////
// MARK: LinearTensorSampler

template<typename T, size_t N>
LinearTensorSampler<T, N>::LinearTensorSampler(const TensorView<const T, N> &view,
                                               const VectorType &gridSpacing,
                                               const VectorType &gridOrigin)
    : _view(view),
      _gridSpacing(gridSpacing),
      _invGridSpacing(ScalarType{1} / gridSpacing),
      _gridOrigin(gridOrigin) {}

template<typename T, size_t N>
LinearTensorSampler<T, N>::LinearTensorSampler(const LinearTensorSampler &other)
    : _view(other._view),
      _gridSpacing(other._gridSpacing),
      _invGridSpacing(other._invGridSpacing),
      _gridOrigin(other._gridOrigin) {}

template<typename T, size_t N>
T LinearTensorSampler<T, N>::operator()(const VectorType &pt) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> shape = _view.shape().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, shape[i], is[i], ts[i]);
    }

    return internal::Lerp<T, N, N>::call(_view, is, ts);
}

template<typename T, size_t N>
void LinearTensorSampler<T, N>::getCoordinatesAndWeights(
    const VectorType &pt, std::array<CoordIndexType, kFlatKernelSize> &indices,
    std::array<ScalarType, kFlatKernelSize> &weights) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> shape = _view.shape().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, shape[i], is[i], ts[i]);
    }

    Vector<size_t, N> viewSize = Vector<size_t, N>::makeConstant(2);
    TensorView<CoordIndexType, N> indexView(indices.data(), viewSize);
    TensorView<ScalarType, N> weightView(weights.data(), viewSize);

    internal::GetCoordinatesAndWeights<ScalarType, N, N>::call(
        indexView, weightView, is.template castTo<size_t>(), ts, 1);
}

template<typename T, size_t N>
void LinearTensorSampler<T, N>::getCoordinatesAndGradientWeights(
    const VectorType &pt, std::array<CoordIndexType, kFlatKernelSize> &indices,
    std::array<VectorType, kFlatKernelSize> &weights) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> shape = _view.shape().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, shape[i], is[i], ts[i]);
    }

    Vector<size_t, N> viewSize = Vector<size_t, N>::makeConstant(2);
    TensorView<CoordIndexType, N> indexView(indices.data(), viewSize);
    TensorView<VectorType, N> weightView(weights.data(), viewSize);

    internal::GetCoordinatesAndGradientWeights<ScalarType, N, N>::call(
        indexView, weightView, is.template castTo<size_t>(), ts,
        _invGridSpacing);
}

template<typename T, size_t N>
std::function<T(const typename LinearTensorSampler<T, N>::VectorType &)>
LinearTensorSampler<T, N>::functor() const {
    LinearTensorSampler sampler(*this);
    return [sampler](const VectorType &x) -> T { return sampler(x); };
}

////////////////////////////////////////////////////////////////////////////////
// MARK: CubicTensorSampler

template<typename T, size_t N, typename CIOp>
CubicTensorSampler<T, N, CIOp>::CubicTensorSampler(
    const TensorView<const T, N> &view, const VectorType &gridSpacing,
    const VectorType &gridOrigin)
    : _view(view),
      _gridSpacing(gridSpacing),
      _invGridSpacing(ScalarType{1} / gridSpacing),
      _gridOrigin(gridOrigin) {}

template<typename T, size_t N, typename CIOp>
CubicTensorSampler<T, N, CIOp>::CubicTensorSampler(const CubicTensorSampler &other)
    : _view(other._view),
      _gridSpacing(other._gridSpacing),
      _invGridSpacing(other._invGridSpacing),
      _gridOrigin(other._gridOrigin) {}

template<typename T, size_t N, typename CIOp>
T CubicTensorSampler<T, N, CIOp>::operator()(const VectorType &pt) const {
    Vector<ssize_t, N> is;
    Vector<ScalarType, N> ts;
    VectorType npt = elemMul(pt - _gridOrigin, _invGridSpacing);
    Vector<ssize_t, N> shape = _view.shape().template castTo<ssize_t>();

    for (size_t i = 0; i < N; ++i) {
        getBarycentric(npt[i], 0, shape[i], is[i], ts[i]);
    }

    return internal::Cubic<T, N, N>::call(_view, is, ts, CIOp());
}

template<typename T, size_t N, typename CIOp>
std::function<T(const typename CubicTensorSampler<T, N, CIOp>::VectorType &)>
CubicTensorSampler<T, N, CIOp>::functor() const {
    CubicTensorSampler sampler(*this);
    return [sampler](const VectorType &x) -> T { return sampler(x); };
}

}// namespace vox