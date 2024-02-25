//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaTensorView

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<T, N>::CudaTensorView() : Base() {}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<T, N>::CudaTensorView(T *ptr, const CudaStdArray<size_t, N> &size_)
    : CudaTensorView() {
    Base::setPtrAndShape(ptr, size_);
}

template<typename T, size_t N>
template<size_t M>
CUDA_CALLABLE CudaTensorView<T, N>::CudaTensorView(
    typename std::enable_if<(M == 1), T>::type *ptr, size_t size_)
    : CudaTensorView(ptr, CudaStdArray<size_t, N>{size_}) {}

template<typename T, size_t N>
CudaTensorView<T, N>::CudaTensorView(CudaTensor<T, N> &other) : CudaTensorView() {
    set(other);
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<T, N>::CudaTensorView(const CudaTensorView &other) {
    set(other);
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<T, N>::CudaTensorView(CudaTensorView &&other) noexcept
    : CudaTensorView() {
    *this = std::move(other);
}

template<typename T, size_t N>
void CudaTensorView<T, N>::set(CudaTensor<T, N> &other) {
    Base::setPtrAndShape(other.data(), other.shape());
}

template<typename T, size_t N>
CUDA_CALLABLE void CudaTensorView<T, N>::set(const CudaTensorView &other) {
    Base::setPtrAndShape(const_cast<T *>(other.data()), other.shape());
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<T, N> &CudaTensorView<T, N>::operator=(
    const CudaTensorView &other) {
    set(other);
    return *this;
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<T, N> &CudaTensorView<T, N>::operator=(
    CudaTensorView &&other) noexcept {
    Base::setPtrAndShape(other.data(), other.shape());
    other.setPtrAndShape(nullptr, CudaStdArray<size_t, N>{});
    return *this;
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::reference
CudaTensorView<T, N>::at(size_t i) {
    return _ptr[i];
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::const_reference
CudaTensorView<T, N>::at(size_t i) const {
    return _ptr[i];
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::reference
CudaTensorView<T, N>::at(size_t i, Args... args) {
    return at(index(i, args...));
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::const_reference
CudaTensorView<T, N>::at(size_t i, Args... args) const {
    return at(index(i, args...));
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::reference
CudaTensorView<T, N>::at(const CudaStdArray<size_t, N> &idx) {
    return at(index(idx));
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::const_reference
CudaTensorView<T, N>::at(const CudaStdArray<size_t, N> &idx) const {
    return at(index(idx));
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::reference
CudaTensorView<T, N>::operator[](size_t i) {
    return at(i);
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::const_reference
CudaTensorView<T, N>::operator[](size_t i) const {
    return at(i);
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::reference
CudaTensorView<T, N>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::const_reference
CudaTensorView<T, N>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::reference
CudaTensorView<T, N>::operator()(const CudaStdArray<size_t, N> &idx) {
    return at(idx);
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<T, N>::Base::const_reference
CudaTensorView<T, N>::operator()(const CudaStdArray<size_t, N> &idx) const {
    return at(idx);
}

////////////////////////////////////////////////////////////////////////////////
// MARK: ConstCudaTensorView

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<const T, N>::CudaTensorView() : Base() {}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<const T, N>::CudaTensorView(const T *ptr, const CudaStdArray<size_t, N> &size_)
    : CudaTensorView() {
    Base::setPtrAndShape(ptr, size_);
}

template<typename T, size_t N>
template<size_t M>
CUDA_CALLABLE CudaTensorView<const T, N>::CudaTensorView(const typename std::enable_if<(M == 1), T>::type *ptr, size_t size_)
    : CudaTensorView(ptr, CudaStdArray<size_t, N>{size_}) {}

template<typename T, size_t N>
CudaTensorView<const T, N>::CudaTensorView(const CudaTensor<T, N> &other)
    : CudaTensorView() {
    set(other);
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<const T, N>::CudaTensorView(const CudaTensorView<T, N> &other) {
    set(other);
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<const T, N>::CudaTensorView(const CudaTensorView &other) {
    set(other);
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<const T, N>::CudaTensorView(CudaTensorView &&other) noexcept
    : CudaTensorView() {
    *this = std::move(other);
}

template<typename T, size_t N>
void CudaTensorView<const T, N>::set(const CudaTensor<T, N> &other) {
    Base::setPtrAndShape(other.data(), other.shape());
}

template<typename T, size_t N>
CUDA_CALLABLE void CudaTensorView<const T, N>::set(const CudaTensorView<T, N> &other) {
    Base::setPtrAndShape(other.data(), other.shape());
}

template<typename T, size_t N>
CUDA_CALLABLE void CudaTensorView<const T, N>::set(const CudaTensorView &other) {
    Base::setPtrAndShape(other.data(), other.shape());
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<const T, N> &CudaTensorView<const T, N>::operator=(const CudaTensorView<T, N> &other) {
    set(other);
    return *this;
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<const T, N> &CudaTensorView<const T, N>::operator=(const CudaTensorView &other) {
    set(other);
    return *this;
}

template<typename T, size_t N>
CUDA_CALLABLE CudaTensorView<const T, N> &CudaTensorView<const T, N>::operator=(CudaTensorView &&other) noexcept {
    Base::setPtrAndShape(other.data(), other.shape());
    other.setPtrAndShape(nullptr, CudaStdArray<size_t, N>{});
    return *this;
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::reference
CudaTensorView<const T, N>::at(size_t i) {
    return _ptr[i];
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::const_reference
CudaTensorView<const T, N>::at(size_t i) const {
    return _ptr[i];
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::reference
CudaTensorView<const T, N>::at(size_t i, Args... args) {
    return at(index(i, args...));
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::const_reference
CudaTensorView<const T, N>::at(size_t i, Args... args) const {
    return at(index(i, args...));
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::reference
CudaTensorView<const T, N>::at(const CudaStdArray<size_t, N> &idx) {
    return at(index(idx));
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::const_reference
CudaTensorView<const T, N>::at(const CudaStdArray<size_t, N> &idx) const {
    return at(index(idx));
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::reference
CudaTensorView<const T, N>::operator[](size_t i) {
    return at(i);
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::const_reference
CudaTensorView<const T, N>::operator[](size_t i) const {
    return at(i);
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::reference
CudaTensorView<const T, N>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template<typename T, size_t N>
template<typename... Args>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::const_reference
CudaTensorView<const T, N>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::reference
CudaTensorView<const T, N>::operator()(const CudaStdArray<size_t, N> &idx) {
    return at(idx);
}

template<typename T, size_t N>
CUDA_CALLABLE_DEVICE typename CudaTensorView<const T, N>::Base::const_reference
CudaTensorView<const T, N>::operator()(const CudaStdArray<size_t, N> &idx) const {
    return at(idx);
}

}// namespace vox
