//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "cuda_tensor_base.h"

namespace vox {
template<typename T, size_t N>
class CudaTensor;

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaTensorView

template<typename T, size_t N>
class CudaTensorView final : public CudaTensorBase<T, N, CudaTensorView<T, N>> {
    using Base = CudaTensorBase<T, N, CudaTensorView<T, N>>;
    using Base::_shape;
    using Base::_ptr;
    using Base::index;
    using Base::setPtrAndShape;

public:
    using Base::data;

    // CTOR
    CUDA_CALLABLE CudaTensorView();

    CUDA_CALLABLE CudaTensorView(T *ptr, const CudaStdArray<size_t, N> &size_);

    template<size_t M = N>
    CUDA_CALLABLE CudaTensorView(typename std::enable_if<(M == 1), T>::type *ptr, size_t size_);

    CudaTensorView(CudaTensor<T, N> &other);

    CUDA_CALLABLE CudaTensorView(const CudaTensorView &other);

    CUDA_CALLABLE CudaTensorView(CudaTensorView &&other) noexcept;

    // set
    void set(CudaTensor<T, N> &other);

    CUDA_CALLABLE void set(const CudaTensorView &other);

    // Assignment Operators
    CUDA_CALLABLE CudaTensorView &operator=(const CudaTensorView &other);

    CUDA_CALLABLE CudaTensorView &operator=(CudaTensorView &&other) noexcept;

public:
    CUDA_CALLABLE_DEVICE Base::reference at(size_t i);

    CUDA_CALLABLE_DEVICE Base::reference at(size_t i) const;

    template<typename... Args>
    CUDA_CALLABLE_DEVICE Base::reference at(size_t i, Args... args);

    template<typename... Args>
    CUDA_CALLABLE_DEVICE Base::reference at(size_t i, Args... args) const;

    CUDA_CALLABLE_DEVICE Base::reference at(const CudaStdArray<size_t, N> &idx);

    CUDA_CALLABLE_DEVICE Base::reference at(const CudaStdArray<size_t, N> &idx) const;

    CUDA_CALLABLE_DEVICE Base::reference operator[](size_t i);

    CUDA_CALLABLE_DEVICE Base::reference operator[](size_t i) const;

    template<typename... Args>
    CUDA_CALLABLE_DEVICE Base::reference operator()(size_t i, Args... args);

    template<typename... Args>
    CUDA_CALLABLE_DEVICE Base::reference operator()(size_t i, Args... args) const;

    CUDA_CALLABLE_DEVICE Base::reference operator()(const CudaStdArray<size_t, N> &idx);

    CUDA_CALLABLE_DEVICE Base::reference
    operator()(const CudaStdArray<size_t, N> &idx) const;
};

////////////////////////////////////////////////////////////////////////////////
// MARK: Immutable CudaTensorView Specialization for CUDA

template<typename T, size_t N>
class CudaTensorView<const T, N> final
    : public CudaTensorBase<const T, N, CudaTensorView<const T, N>> {
    using Base = CudaTensorBase<const T, N, CudaTensorView<const T, N>>;
    using Base::_shape;
    using Base::_ptr;
    using Base::index;
    using Base::setPtrAndShape;

public:
    using Base::data;

    // CTOR
    CUDA_CALLABLE CudaTensorView();

    CUDA_CALLABLE CudaTensorView(const T *ptr, const CudaStdArray<size_t, N> &size_);

    template<size_t M = N>
    CUDA_CALLABLE CudaTensorView(const typename std::enable_if<(M == 1), T>::type *ptr, size_t size_);

    CudaTensorView(const CudaTensor<T, N> &other);

    CUDA_CALLABLE CudaTensorView(const CudaTensorView<T, N> &other);

    CUDA_CALLABLE CudaTensorView(const CudaTensorView &other);

    CUDA_CALLABLE CudaTensorView(CudaTensorView &&) noexcept;

    // set
    void set(const CudaTensor<T, N> &other);

    CUDA_CALLABLE void set(const CudaTensorView<T, N> &other);

    CUDA_CALLABLE void set(const CudaTensorView &other);

    // Assignment Operators
    CUDA_CALLABLE CudaTensorView &operator=(const CudaTensorView<T, N> &other);

    CUDA_CALLABLE CudaTensorView &operator=(const CudaTensorView &other);

    CUDA_CALLABLE CudaTensorView &operator=(CudaTensorView &&other) noexcept;

public:
    CUDA_CALLABLE_DEVICE Base::reference at(size_t i);

    CUDA_CALLABLE_DEVICE Base::const_reference at(size_t i) const;

    template<typename... Args>
    CUDA_CALLABLE_DEVICE Base::reference at(size_t i, Args... args);

    template<typename... Args>
    CUDA_CALLABLE_DEVICE Base::const_reference at(size_t i, Args... args) const;

    CUDA_CALLABLE_DEVICE Base::reference at(const CudaStdArray<size_t, N> &idx);

    CUDA_CALLABLE_DEVICE Base::const_reference at(const CudaStdArray<size_t, N> &idx) const;

    CUDA_CALLABLE_DEVICE Base::reference operator[](size_t i);

    CUDA_CALLABLE_DEVICE Base::const_reference operator[](size_t i) const;

    template<typename... Args>
    CUDA_CALLABLE_DEVICE Base::reference operator()(size_t i, Args... args);

    template<typename... Args>
    CUDA_CALLABLE_DEVICE Base::const_reference operator()(size_t i, Args... args) const;

    CUDA_CALLABLE_DEVICE Base::reference operator()(const CudaStdArray<size_t, N> &idx);

    CUDA_CALLABLE_DEVICE Base::const_reference
    operator()(const CudaStdArray<size_t, N> &idx) const;
};

template<class T>
using CudaTensorView1 = CudaTensorView<T, 1>;

template<class T>
using CudaTensorView2 = CudaTensorView<T, 2>;

template<class T>
using CudaTensorView3 = CudaTensorView<T, 3>;

template<class T>
using CudaTensorView4 = CudaTensorView<T, 4>;

template<class T>
using ConstCudaTensorView1 = CudaTensorView<const T, 1>;

template<class T>
using ConstCudaTensorView2 = CudaTensorView<const T, 2>;

template<class T>
using ConstCudaTensorView3 = CudaTensorView<const T, 3>;

template<class T>
using ConstCudaTensorView4 = CudaTensorView<const T, 4>;

}// namespace vox

#include "cuda_tensor_view-inl.h"
