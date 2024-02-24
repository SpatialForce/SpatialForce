//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/cuda_std_array.h"

namespace vox {

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaTensorBase

template<typename T, size_t N, typename DerivedTensor>
class CudaTensorBase {
public:
    using Derived = DerivedTensor;
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;

    CUDA_CALLABLE size_t index(size_t i) const;

    template<typename... Args>
    CUDA_CALLABLE size_t index(size_t i, Args... args) const;

    template<size_t... I>
    CUDA_CALLABLE size_t index(const CudaStdArray<size_t, N> &idx) const;

    CUDA_CALLABLE T *data();

    CUDA_CALLABLE const T *data() const;

    CUDA_CALLABLE const CudaStdArray<size_t, N> &shape() const;

    template<size_t M = N>
    CUDA_CALLABLE std::enable_if_t<(M > 0), size_t> width() const;

    template<size_t M = N>
    CUDA_CALLABLE std::enable_if_t<(M > 1), size_t> height() const;

    template<size_t M = N>
    CUDA_CALLABLE std::enable_if_t<(M > 2), size_t> depth() const;

    CUDA_CALLABLE size_t length() const;

    CUDA_CALLABLE reference at(size_t i);

    CUDA_CALLABLE const_reference at(size_t i) const;

    template<typename... Args>
    CUDA_CALLABLE reference at(size_t i, Args... args);

    template<typename... Args>
    CUDA_CALLABLE const_reference at(size_t i, Args... args) const;

    CUDA_CALLABLE reference at(const CudaStdArray<size_t, N> &idx);

    CUDA_CALLABLE const_reference at(const CudaStdArray<size_t, N> &idx) const;

    CUDA_CALLABLE reference operator[](size_t i);

    CUDA_CALLABLE const_reference operator[](size_t i) const;

    template<typename... Args>
    CUDA_CALLABLE reference operator()(size_t i, Args... args);

    template<typename... Args>
    CUDA_CALLABLE const_reference operator()(size_t i, Args... args) const;

    CUDA_CALLABLE reference operator()(const CudaStdArray<size_t, N> &idx);

    CUDA_CALLABLE const_reference operator()(const CudaStdArray<size_t, N> &idx) const;

protected:
    pointer _ptr = nullptr;
    CudaStdArray<size_t, N> _shape;

    CUDA_CALLABLE CudaTensorBase();

    CUDA_CALLABLE CudaTensorBase(const CudaTensorBase &other);

    CUDA_CALLABLE CudaTensorBase(CudaTensorBase &&other);

    template<typename... Args>
    CUDA_CALLABLE void setPtrAndShape(pointer ptr, size_t ni, Args... args);

    CUDA_CALLABLE void setPtrAndShape(pointer data, CudaStdArray<size_t, N> shape);

    CUDA_CALLABLE void swapPtrAndShape(CudaTensorBase &other);

    CUDA_CALLABLE void clearPtrAndShape();

    CUDA_CALLABLE CudaTensorBase &operator=(const CudaTensorBase &other);

    CUDA_CALLABLE CudaTensorBase &operator=(CudaTensorBase &&other);

private:
    template<typename... Args>
    CUDA_CALLABLE size_t _index(size_t d, size_t i, Args... args) const;

    CUDA_CALLABLE size_t _index(size_t, size_t i) const;

    template<size_t... I>
    CUDA_CALLABLE size_t _index(const CudaStdArray<size_t, N> &idx,
                                std::index_sequence<I...>) const;
};

}// namespace vox

#include "cuda_tensor_base-inl.h"
