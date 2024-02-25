//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/define.h"

namespace vox {
template<typename T, size_t Rows>
using Vector = Matrix<T, Rows, 1>;

template<typename T, size_t N>
class CudaStdArray {
public:
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;
    using iterator = pointer;
    using const_iterator = const_pointer;

    CUDA_CALLABLE CudaStdArray();

    template<typename... Args>
    CUDA_CALLABLE explicit constexpr CudaStdArray(const_reference first, Args... rest);

    CUDA_CALLABLE constexpr CudaStdArray(const std::initializer_list<T> &lst);

    CUDA_CALLABLE explicit constexpr CudaStdArray(const Vector<T, N> &other);

    CUDA_CALLABLE constexpr CudaStdArray(const CudaStdArray &other);

    CUDA_CALLABLE constexpr void fill(const_reference val);

    CUDA_CALLABLE constexpr Vector<T, N> toVector() const;

    CUDA_CALLABLE constexpr reference operator[](size_t i);

    CUDA_CALLABLE constexpr const_reference operator[](size_t i) const;

    CUDA_CALLABLE constexpr bool operator==(const CudaStdArray &other) const;

    CUDA_CALLABLE constexpr bool operator!=(const CudaStdArray &other) const;

    CUDA_CALLABLE constexpr size_t size() const {
        return N;
    }

private:
    T _elements[N];

    template<typename... Args>
    CUDA_CALLABLE constexpr void setAt(size_t i, const_reference first, Args... rest);

    template<typename... Args>
    CUDA_CALLABLE constexpr void setAt(size_t i, const_reference first);
};

}// namespace vox

#include "cuda_std_array-inl.h"
