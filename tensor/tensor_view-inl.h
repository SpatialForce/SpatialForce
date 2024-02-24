//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {

// MARK: TensorView

template<typename T, size_t N>
TensorView<T, N>::TensorView() : Base() {}

template<typename T, size_t N>
TensorView<T, N>::TensorView(T *ptr, const Vector<size_t, N> &size_)
    : TensorView() {
    Base::setPtrAndSize(ptr, size_);
}

template<typename T, size_t N>
template<size_t M>
TensorView<T, N>::TensorView(typename std::enable_if_t<(M == 1), T *> ptr,
                             size_t size_)
    : TensorView(ptr, Vector<size_t, N>{size_}) {}

template<typename T, size_t N>
TensorView<T, N>::TensorView(Tensor<T, N> &other) : TensorView() {
    set(other);
}

template<typename T, size_t N>
TensorView<T, N>::TensorView(const TensorView &other) {
    set(other);
}

template<typename T, size_t N>
TensorView<T, N>::TensorView(TensorView &&other) noexcept : TensorView() {
    *this = std::move(other);
}

template<typename T, size_t N>
void TensorView<T, N>::set(Tensor<T, N> &other) {
    Base::setPtrAndSize(other.data(), other.shape());
}

template<typename T, size_t N>
void TensorView<T, N>::set(const TensorView &other) {
    Base::setPtrAndSize(const_cast<T *>(other.data()), other.shape());
}

template<typename T, size_t N>
void TensorView<T, N>::fill(const T &val) {
    forEachIndex(Vector<size_t, N>{}, _shape,
                 [&](auto... idx) { this->at(idx...) = val; });
}

template<typename T, size_t N>
TensorView<T, N> &TensorView<T, N>::operator=(const TensorView &other) {
    set(other);
    return *this;
}

template<typename T, size_t N>
TensorView<T, N> &TensorView<T, N>::operator=(TensorView &&other) noexcept {
    Base::setPtrAndSize(other.data(), other.shape());
    other.setPtrAndSize(nullptr, Vector<size_t, N>{});
    return *this;
}

// MARK: ConstTensorView

template<typename T, size_t N>
TensorView<const T, N>::TensorView() : Base() {}

template<typename T, size_t N>
TensorView<const T, N>::TensorView(const T *ptr, const Vector<size_t, N> &size_)
    : TensorView() {
    Base::setPtrAndSize(ptr, size_);
}

template<typename T, size_t N>
template<size_t M>
TensorView<const T, N>::TensorView(
    typename std::enable_if_t<(M == 1), const T *> ptr, size_t size_)
    : TensorView(ptr, Vector<size_t, N>{size_}) {}

template<typename T, size_t N>
TensorView<const T, N>::TensorView(const Tensor<T, N> &other) : TensorView() {
    set(other);
}

template<typename T, size_t N>
TensorView<const T, N>::TensorView(const TensorView<T, N> &other) {
    set(other);
}

template<typename T, size_t N>
TensorView<const T, N>::TensorView(const TensorView<const T, N> &other) {
    set(other);
}

template<typename T, size_t N>
TensorView<const T, N>::TensorView(TensorView &&other) noexcept : TensorView() {
    *this = std::move(other);
}

template<typename T, size_t N>
void TensorView<const T, N>::set(const Tensor<T, N> &other) {
    Base::setPtrAndSize(other.data(), other.shape());
}

template<typename T, size_t N>
void TensorView<const T, N>::set(const TensorView<T, N> &other) {
    Base::setPtrAndSize(other.data(), other.shape());
}

template<typename T, size_t N>
void TensorView<const T, N>::set(const TensorView<const T, N> &other) {
    Base::setPtrAndSize(other.data(), other.shape());
}

template<typename T, size_t N>
TensorView<const T, N> &TensorView<const T, N>::operator=(
    const TensorView<T, N> &other) {
    set(other);
    return *this;
}

template<typename T, size_t N>
TensorView<const T, N> &TensorView<const T, N>::operator=(
    const TensorView<const T, N> &other) {
    set(other);
    return *this;
}

template<typename T, size_t N>
TensorView<const T, N> &TensorView<const T, N>::operator=(
    TensorView<const T, N> &&other) noexcept {
    Base::setPtrAndSize(other.data(), other.shape());
    other.setPtrAndSize(nullptr, Vector<size_t, N>{});
    return *this;
}

}// namespace vox