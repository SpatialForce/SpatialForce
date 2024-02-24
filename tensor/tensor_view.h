//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/matrix.h"

namespace vox {

template<typename T, size_t N, typename Derived>
class TensorBase;

template<typename T, size_t N>
class Tensor;

// MARK: TensorView

template<typename T, size_t N>
class TensorView final : public TensorBase<T, N, TensorView<T, N>> {
    using Base = TensorBase<T, N, TensorView<T, N>>;
    using Base::_shape;
    using Base::setPtrAndShape;
    using Base::at;

public:
    // CTOR
    TensorView();

    TensorView(T *ptr, const Vector<size_t, N> &shape_);

    template<size_t M = N>
    TensorView(typename std::enable_if_t<(M == 1), T *> ptr, size_t size_);

    TensorView(Tensor<T, N> &other);

    TensorView(const TensorView &other);

    TensorView(TensorView &&other) noexcept;

    // set

    void set(Tensor<T, N> &other);

    void set(const TensorView &other);

    void fill(const T &val);

    // Assignment Operators
    TensorView &operator=(const TensorView &other);

    TensorView &operator=(TensorView &&other) noexcept;
};

template<typename T, size_t N>
class TensorView<const T, N> final
    : public TensorBase<const T, N, TensorView<const T, N>> {
    using Base = TensorBase<const T, N, TensorView<const T, N>>;
    using Base::_shape;
    using Base::setPtrAndShape;

public:
    // CTOR
    TensorView();

    TensorView(const T *ptr, const Vector<size_t, N> &shape_);

    template<size_t M = N>
    TensorView(typename std::enable_if_t<(M == 1), const T *> ptr, size_t size_);

    TensorView(const Tensor<T, N> &other);

    TensorView(const TensorView<T, N> &other);

    TensorView(const TensorView<const T, N> &other);

    TensorView(TensorView<const T, N> &&) noexcept;

    // set

    void set(const Tensor<T, N> &other);

    void set(const TensorView<T, N> &other);

    void set(const TensorView<const T, N> &other);

    // Assignment Operators
    TensorView &operator=(const TensorView<T, N> &other);

    TensorView &operator=(const TensorView<const T, N> &other);

    TensorView &operator=(TensorView<const T, N> &&other) noexcept;
};

template<class T>
using TensorView1 = TensorView<T, 1>;

template<class T>
using TensorView2 = TensorView<T, 2>;

template<class T>
using TensorView3 = TensorView<T, 3>;

template<class T>
using TensorView4 = TensorView<T, 4>;

template<class T>
using ConstTensorView1 = TensorView<const T, 1>;

template<class T>
using ConstTensorView2 = TensorView<const T, 2>;

template<class T>
using ConstTensorView3 = TensorView<const T, 3>;

template<class T>
using ConstTensorView4 = TensorView<const T, 4>;

}// namespace vox

#include "tensor_view-inl.h"