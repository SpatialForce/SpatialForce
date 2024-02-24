//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {
namespace {
template<typename T>
CUDA_CALLABLE inline void cudaSwap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}
}// namespace
////////////////////////////////////////////////////////////////////////////////
// MARK: CudaTensorBase

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE size_t CudaTensorBase<T, N, Derived>::index(size_t i) const {
    return i;
}

template<typename T, size_t N, typename Derived>
template<typename... Args>
CUDA_CALLABLE size_t CudaTensorBase<T, N, Derived>::index(size_t i, Args... args) const {
    static_assert(sizeof...(args) == N - 1, "Invalid number of indices.");
    return i + _shape[0] * _index(1, args...);
}

template<typename T, size_t N, typename Derived>
template<size_t... I>
CUDA_CALLABLE size_t CudaTensorBase<T, N, Derived>::index(
    const CudaStdArray<size_t, N> &idx) const {
    return _index(idx, std::make_index_sequence<N>{});
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE T *CudaTensorBase<T, N, Derived>::data() {
    return _ptr;
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE const T *CudaTensorBase<T, N, Derived>::data() const {
    return _ptr;
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE const CudaStdArray<size_t, N> &CudaTensorBase<T, N, Derived>::shape() const {
    return _shape;
}

template<typename T, size_t N, typename Derived>
template<size_t M>
CUDA_CALLABLE std::enable_if_t<(M > 0), size_t> CudaTensorBase<T, N, Derived>::width() const {
    return _shape[0];
}

template<typename T, size_t N, typename Derived>
template<size_t M>
CUDA_CALLABLE std::enable_if_t<(M > 1), size_t> CudaTensorBase<T, N, Derived>::height() const {
    return _shape[1];
}

template<typename T, size_t N, typename Derived>
template<size_t M>
CUDA_CALLABLE std::enable_if_t<(M > 2), size_t> CudaTensorBase<T, N, Derived>::depth() const {
    return _shape[2];
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE size_t CudaTensorBase<T, N, Derived>::length() const {
    // TODO: Replace CudaStdArray with Vector
    // return product<size_t, N>(_shape, 1);
    size_t l = _shape[0];
    for (size_t i = 1; i < N; ++i) {
        l *= _shape[i];
    }
    return l;
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::reference
CudaTensorBase<T, N, Derived>::at(size_t i) {
    return _ptr[i];
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::const_reference
CudaTensorBase<T, N, Derived>::at(size_t i) const {
    return _ptr[i];
}

template<typename T, size_t N, typename Derived>
template<typename... Args>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::reference
CudaTensorBase<T, N, Derived>::at(size_t i, Args... args) {
    return at(index(i, args...));
}

template<typename T, size_t N, typename Derived>
template<typename... Args>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::const_reference
CudaTensorBase<T, N, Derived>::at(size_t i, Args... args) const {
    return at(index(i, args...));
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::reference
CudaTensorBase<T, N, Derived>::at(const CudaStdArray<size_t, N> &idx) {
    return at(index(idx));
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::const_reference
CudaTensorBase<T, N, Derived>::at(const CudaStdArray<size_t, N> &idx) const {
    return at(index(idx));
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::reference
CudaTensorBase<T, N, Derived>::operator[](size_t i) {
    return at(i);
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::const_reference
CudaTensorBase<T, N, Derived>::operator[](size_t i) const {
    return at(i);
}

template<typename T, size_t N, typename Derived>
template<typename... Args>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::reference
CudaTensorBase<T, N, Derived>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template<typename T, size_t N, typename Derived>
template<typename... Args>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::const_reference
CudaTensorBase<T, N, Derived>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::reference
CudaTensorBase<T, N, Derived>::operator()(const CudaStdArray<size_t, N> &idx) {
    return at(idx);
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE typename CudaTensorBase<T, N, Derived>::const_reference
CudaTensorBase<T, N, Derived>::operator()(
    const CudaStdArray<size_t, N> &idx) const {
    return at(idx);
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE CudaTensorBase<T, N, Derived>::CudaTensorBase() : _shape{} {}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE CudaTensorBase<T, N, Derived>::CudaTensorBase(const CudaTensorBase &other) {
    setPtrAndShape(other._ptr, other._shape);
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE CudaTensorBase<T, N, Derived>::CudaTensorBase(CudaTensorBase &&other) {
    *this = std::move(other);
}

template<typename T, size_t N, typename Derived>
template<typename... Args>
CUDA_CALLABLE void CudaTensorBase<T, N, Derived>::setPtrAndShape(pointer ptr, size_t ni, Args... args) {
    setPtrAndShape(ptr, CudaStdArray<size_t, N>{ni, args...});
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE void CudaTensorBase<T, N, Derived>::setPtrAndShape(pointer ptr, CudaStdArray<size_t, N> shape) {
    _ptr = ptr;
    _shape = shape;
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE void CudaTensorBase<T, N, Derived>::swapPtrAndShape(CudaTensorBase &other) {
    cudaSwap(_ptr, other._ptr);
    cudaSwap(_shape, other._shape);
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE void CudaTensorBase<T, N, Derived>::clearPtrAndShape() {
    setPtrAndShape(nullptr, CudaStdArray<size_t, N>{});
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE CudaTensorBase<T, N, Derived> &CudaTensorBase<T, N, Derived>::operator=(
    const CudaTensorBase &other) {
    setPtrAndShape(other._ptr, other._shape);
    return *this;
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE CudaTensorBase<T, N, Derived> &CudaTensorBase<T, N, Derived>::operator=(
    CudaTensorBase &&other) {
    setPtrAndShape(other._ptr, other._shape);
    other.setPtrAndShape(nullptr, CudaStdArray<size_t, N>{});
    return *this;
}

template<typename T, size_t N, typename Derived>
template<typename... Args>
CUDA_CALLABLE size_t CudaTensorBase<T, N, Derived>::_index(size_t d, size_t i, Args... args) const {
    return i + _shape[d] * _index(d + 1, args...);
}

template<typename T, size_t N, typename Derived>
CUDA_CALLABLE size_t CudaTensorBase<T, N, Derived>::_index(size_t, size_t i) const {
    return i;
}

template<typename T, size_t N, typename Derived>
template<size_t... I>
CUDA_CALLABLE size_t CudaTensorBase<T, N, Derived>::_index(const CudaStdArray<size_t, N> &idx, std::index_sequence<I...>) const {
    return index(idx[I]...);
}

}// namespace vox
