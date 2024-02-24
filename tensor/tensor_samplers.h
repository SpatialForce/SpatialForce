//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "tensor_view.h"
#include "math/math_utils.h"
#include "math/matrix.h"
#include "type_helpers.h"
#include <functional>

namespace vox {

////////////////////////////////////////////////////////////////////////////////
// MARK: NearestTensorSampler

//!
//! \brief N-D nearest tensor sampler class.
//!
//! This class provides nearest sampling interface for a given N-D tensor.
//!
//! \tparam T - The value type to sample.
//! \tparam N - Dimension.
//!
template<typename T, size_t N>
class NearestTensorSampler final {
public:
    static_assert(N > 0, "Dimension should be greater than 0");

    using ScalarType = typename GetScalarType<T>::value;

    static_assert(std::is_floating_point<ScalarType>::value,
                  "NearestTensorSampler only can be instantiated with floating "
                  "point types");

    using VectorType = Vector<ScalarType, N>;
    using CoordIndexType = Vector<size_t, N>;

    NearestTensorSampler() = default;

    //!
    //! \brief      Constructs a sampler.
    //!
    //! \param[in]  view        The tensor view.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  gridOrigin  The grid origin.
    //!
    explicit NearestTensorSampler(const TensorView<const T, N> &view,
                                  const VectorType &gridSpacing,
                                  const VectorType &gridOrigin);

    //! Copy constructor.
    NearestTensorSampler(const NearestTensorSampler &other);

    //! Returns sampled value at point \p pt.
    T operator()(const VectorType &pt) const;

    //! Returns the nearest tensor index for point \p x.
    CoordIndexType getCoordinate(const VectorType &pt) const;

    //! Returns a funtion object that wraps this instance.
    std::function<T(const VectorType &)> functor() const;

private:
    TensorView<const T, N> _view;
    VectorType _gridSpacing;
    VectorType _invGridSpacing;
    VectorType _gridOrigin;
};

template<typename T>
using NearestTensorSampler1 = NearestTensorSampler<T, 1>;

template<typename T>
using NearestTensorSampler2 = NearestTensorSampler<T, 2>;

template<typename T>
using NearestTensorSampler3 = NearestTensorSampler<T, 3>;

////////////////////////////////////////////////////////////////////////////////
// MARK: LinearTensorSampler

//!
//! \brief N-D tensor sampler using linear interpolation.
//!
//! This class provides linear sampling interface for a given N-D tensor.
//!
//! \tparam T - The value type to sample.
//! \tparam N - Dimension.
//!
template<typename T, size_t N>
class LinearTensorSampler final {
public:
    static_assert(N > 0, "N should be greater than 0");

    using ScalarType = typename GetScalarType<T>::value;

    static_assert(std::is_floating_point<ScalarType>::value,
                  "LinearTensorSampler only can be instantiated with floating "
                  "point types");

    using VectorType = Vector<ScalarType, N>;
    using CoordIndexType = Vector<size_t, N>;

    static constexpr size_t kFlatKernelSize = 1 << N;

    LinearTensorSampler() = default;

    //!
    //! \brief      Constructs a sampler.
    //!
    //! \param[in]  view        The tensor view.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  gridOrigin  The grid origin.
    //!
    explicit LinearTensorSampler(const TensorView<const T, N> &view,
                                 const VectorType &gridSpacing,
                                 const VectorType &gridOrigin);

    //! Copy constructor.
    LinearTensorSampler(const LinearTensorSampler &other);

    //! Returns sampled value at point \p pt.
    T operator()(const VectorType &pt) const;

    //! Returns the indices of points and their sampling weight for given point.
    void getCoordinatesAndWeights(
        const VectorType &pt,
        std::array<CoordIndexType, kFlatKernelSize> &indices,
        std::array<ScalarType, kFlatKernelSize> &weights) const;

    //! Returns the indices of points and their gradient of sampling weight for
    //! given point.
    void getCoordinatesAndGradientWeights(
        const VectorType &pt,
        std::array<CoordIndexType, kFlatKernelSize> &indices,
        std::array<VectorType, kFlatKernelSize> &weights) const;

    //! Returns a std::function instance that wraps this instance.
    std::function<T(const VectorType &)> functor() const;

private:
    TensorView<const T, N> _view;
    VectorType _gridSpacing;
    VectorType _invGridSpacing;
    VectorType _gridOrigin;
};

template<typename T>
using LinearTensorSampler1 = LinearTensorSampler<T, 1>;

template<typename T>
using LinearTensorSampler2 = LinearTensorSampler<T, 2>;

template<typename T>
using LinearTensorSampler3 = LinearTensorSampler<T, 3>;

////////////////////////////////////////////////////////////////////////////////
// MARK: CubicTensorSampler

//!
//! \brief N-D cubic tensor sampler class.
//!
//! This class provides cubic sampling interface for a given N-D tensor.
//!
//! \tparam T - The value type to sample.
//! \tparam N - Dimension.
//!
template<typename T, size_t N, typename CubicInterpolationOp>
class CubicTensorSampler final {
public:
    static_assert(N > 0, "N should be greater than 0");

    using ScalarType = typename GetScalarType<T>::value;

    static_assert(std::is_floating_point<ScalarType>::value,
                  "CubicTensorSampler only can be instantiated with floating "
                  "point types");

    using VectorType = Vector<ScalarType, N>;
    using CoordIndexType = Vector<size_t, N>;

    CubicTensorSampler() = default;

    //!
    //! \brief      Constructs a sampler.
    //!
    //! \param[in]  view        The tensor view.
    //! \param[in]  gridSpacing The grid spacing.
    //! \param[in]  gridOrigin  The grid origin.
    //!
    explicit CubicTensorSampler(const TensorView<const T, N> &view,
                                const VectorType &gridSpacing,
                                const VectorType &gridOrigin);

    //! Copy constructor.
    CubicTensorSampler(const CubicTensorSampler &other);

    //! Returns sampled value at point \p pt.
    T operator()(const VectorType &pt) const;

    //! Returns a funtion object that wraps this instance.
    std::function<T(const VectorType &)> functor() const;

private:
    TensorView<const T, N> _view;
    VectorType _gridSpacing;
    VectorType _invGridSpacing;
    VectorType _gridOrigin;
};

template<typename T>
struct CatmullRom {
    using ScalarType = typename GetScalarType<T>::value;

    T operator()(const T &f0, const T &f1, const T &f2, const T &f3,
                 ScalarType t) const {
        return catmullRom(f0, f1, f2, f3, t);
    }
};

template<typename T>
struct MonotonicCatmullRom {
    using ScalarType = typename GetScalarType<T>::value;

    T operator()(const T &f0, const T &f1, const T &f2, const T &f3,
                 ScalarType t) const {
        return monotonicCatmullRom(f0, f1, f2, f3, t);
    }
};

template<typename T>
using CatmullRomTensorSampler1 =
    CubicTensorSampler<T, 1, CatmullRom<T>>;

template<typename T>
using CatmullRomTensorSampler2 =
    CubicTensorSampler<T, 2, CatmullRom<T>>;

template<typename T>
using CatmullRomTensorSampler3 =
    CubicTensorSampler<T, 3, CatmullRom<T>>;

template<typename T>
using MonotonicCatmullRomTensorSampler1 =
    CubicTensorSampler<T, 1, MonotonicCatmullRom<T>>;

template<typename T>
using MonotonicCatmullRomTensorSampler2 =
    CubicTensorSampler<T, 2, MonotonicCatmullRom<T>>;

template<typename T>
using MonotonicCatmullRomTensorSampler3 =
    CubicTensorSampler<T, 3, MonotonicCatmullRom<T>>;

}// namespace vox

#include "tensor_samplers-inl.h"