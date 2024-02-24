//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {

template<typename T, size_t N>
CUDA_CALLABLE CudaStdArray<T, N>::CudaStdArray() {
    fill(T{});
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE CudaStdArray<T, N>::CudaStdArray(const_reference first, Args... rest) {
    static_assert(
        sizeof...(Args) == N - 1,
        "Number of arguments should be equal to the size of the vector.");
    setAt(0, first, rest...);
}

template<typename T, size_t N>
CUDA_CALLABLE CudaStdArray<T, N>::CudaStdArray(const std::initializer_list<T> &lst) {
    auto iter = lst.begin();
    for (size_t i = 0; i < N; ++i) {
        _elements[i] = *(iter++);
    }
}

template<typename T, size_t N>
CUDA_CALLABLE CudaStdArray<T, N>::CudaStdArray(const Vector<T, N> &other) {
    for (size_t i = 0; i < N; ++i) {
        _elements[i] = other[i];
    }
}

template<typename T, size_t N>
CUDA_CALLABLE CudaStdArray<T, N>::CudaStdArray(const CudaStdArray &other) {
    for (size_t i = 0; i < N; ++i) {
        _elements[i] = other[i];
    }
}

template<typename T, size_t N>
CUDA_CALLABLE void CudaStdArray<T, N>::fill(const_reference val) {
    for (size_t i = 0; i < N; ++i) {
        _elements[i] = val;
    }
}

template<typename T, size_t N>
CUDA_CALLABLE Vector<T, N> CudaStdArray<T, N>::toVector() const {
    Vector<T, N> vec;
    for (size_t i = 0; i < N; ++i) {
        vec[i] = _elements[i];
    }
    return vec;
}

template<typename T, size_t N>
CUDA_CALLABLE typename CudaStdArray<T, N>::reference CudaStdArray<T, N>::operator[](
    size_t i) {
    return _elements[i];
}

template<typename T, size_t N>
CUDA_CALLABLE typename CudaStdArray<T, N>::const_reference CudaStdArray<T, N>::operator[](
    size_t i) const {
    return _elements[i];
}

template<typename T, size_t N>
CUDA_CALLABLE bool CudaStdArray<T, N>::operator==(const CudaStdArray &other) const {
    for (size_t i = 0; i < N; ++i) {
        if (_elements[i] != other._elements[i]) {
            return false;
        }
    }

    return true;
}

template<typename T, size_t N>
CUDA_CALLABLE bool CudaStdArray<T, N>::operator!=(const CudaStdArray &other) const {
    return !(*this == other);
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE void CudaStdArray<T, N>::setAt(size_t i, const_reference first, Args... rest) {
    _elements[i] = first;
    setAt(i + 1, rest...);
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE void CudaStdArray<T, N>::setAt(size_t i, const_reference first) {
    _elements[i] = first;
}

}// namespace vox
