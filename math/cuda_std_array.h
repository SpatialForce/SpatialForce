//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "core/define.h"

namespace vox {

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
    CUDA_CALLABLE CudaStdArray(const_reference first, Args... rest);

    CUDA_CALLABLE CudaStdArray(const std::initializer_list<T> &lst);

    CUDA_CALLABLE CudaStdArray(const CudaStdArray &other);

    CUDA_CALLABLE void fill(const_reference val);

    CUDA_CALLABLE reference operator[](size_t i);

    CUDA_CALLABLE const_reference operator[](size_t i) const;

    CUDA_CALLABLE bool operator==(const CudaStdArray &other) const;

    CUDA_CALLABLE bool operator!=(const CudaStdArray &other) const;

private:
    T _elements[N];

    template<typename... Args>
    CUDA_CALLABLE void setAt(size_t i, const_reference first, Args... rest);

    template<typename... Args>
    CUDA_CALLABLE void setAt(size_t i, const_reference first);
};

}// namespace vox

#include "cuda_std_array-inl.h"
