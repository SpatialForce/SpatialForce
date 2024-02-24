//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "tensor_view.h"
#include "iteration_utils.h"

namespace vox {

template<typename T, size_t N>
class Tensor;

namespace internal {

template<typename T, size_t N, size_t I>
struct GetSizeAndInitVal {
    template<typename... Args>
    static void call(Vector<size_t, N> &size, T &value, size_t n,
                     Args... args) {
        size[N - I - 1] = n;
        GetSizeAndInitVal<T, N, I - 1>::call(size, value, args...);
    }
};

template<typename T, size_t N>
struct GetSizeAndInitVal<T, N, 0> {
    static void call(Vector<size_t, N> &size, T &value, size_t n) {
        call(size, value, n, T{});
    }

    static void call(Vector<size_t, N> &size, T &value, size_t n,
                     const T &initVal) {
        size[N - 1] = n;
        value = initVal;
    }
};

template<typename T, size_t N, size_t I>
struct GetSizeFromInitList {
    static size_t call(Vector<size_t, N> &size,
                       NestedInitializerListsT<T, I> lst) {
        size[I - 1] = lst.size();
        size_t i = 0;
        for (auto subLst : lst) {
            if (i == 0) {
                GetSizeFromInitList<T, N, I - 1>::call(size, subLst);
            } else {
                Vector<size_t, N> tempSizeN;
                size_t otherSize =
                    GetSizeFromInitList<T, N, I - 1>::call(tempSizeN, subLst);
                (void)otherSize;
                ASSERT(otherSize == tempSizeN[I - 2]);
            }
            ++i;
        }
        return size[I - 1];
    }
};

template<typename T, size_t N>
struct GetSizeFromInitList<T, N, 1> {
    static size_t call(Vector<size_t, N> &size,
                       NestedInitializerListsT<T, 1> lst) {
        size[0] = lst.size();
        return size[0];
    }
};

template<typename T, size_t N, size_t I>
struct SetTensorFromInitList {
    static void call(Tensor<T, N> &arr, NestedInitializerListsT<T, I> lst) {
        size_t i = 0;
        for (auto subLst : lst) {
            ASSERT(i < arr.shape()[I - 1]);
            SetTensorFromInitList<T, N, I - 1>::call(arr, subLst, i);
            ++i;
        }
    }

    template<typename... RemainingIndices>
    static void call(Tensor<T, N> &arr, NestedInitializerListsT<T, I> lst,
                     RemainingIndices... indices) {
        size_t i = 0;
        for (auto subLst : lst) {
            ASSERT(i < arr.shape()[I - 1]);
            SetTensorFromInitList<T, N, I - 1>::call(arr, subLst, i, indices...);
            ++i;
        }
    }
};

template<typename T, size_t N>
struct SetTensorFromInitList<T, N, 1> {
    static void call(Tensor<T, N> &arr, NestedInitializerListsT<T, 1> lst) {
        size_t i = 0;
        for (auto val : lst) {
            ASSERT(i < arr.shape()[0]);
            arr(i) = val;
            ++i;
        }
    }

    template<typename... RemainingIndices>
    static void call(Tensor<T, N> &arr, NestedInitializerListsT<T, 1> lst,
                     RemainingIndices... indices) {
        size_t i = 0;
        for (auto val : lst) {
            ASSERT(i < arr.shape()[0]);
            arr(i, indices...) = val;
            ++i;
        }
    }
};

}// namespace internal

// MARK: TensorBase
template<typename T, size_t N, typename D>
size_t TensorBase<T, N, D>::index(size_t i) const {
    return i;
}

template<typename T, size_t N, typename D>
template<typename... Args>
size_t TensorBase<T, N, D>::index(size_t i, Args... args) const {
    static_assert(sizeof...(args) == N - 1, "Invalid number of indices.");
    return i + _shape[0] * _index(1, args...);
}

template<typename T, size_t N, typename D>
template<size_t... I>
size_t TensorBase<T, N, D>::index(const Vector<size_t, N> &idx) const {
    return _index(idx, std::make_index_sequence<N>{});
}

template<typename T, size_t N, typename D>
T *TensorBase<T, N, D>::data() {
    return _ptr;
}

template<typename T, size_t N, typename D>
const T *TensorBase<T, N, D>::data() const {
    return _ptr;
}

template<typename T, size_t N, typename D>
const Vector<size_t, N> &TensorBase<T, N, D>::shape() const {
    return _shape;
}

template<typename T, size_t N, typename D>
template<size_t M>
std::enable_if_t<(M > 0), size_t> TensorBase<T, N, D>::width() const {
    return _shape.x;
}

template<typename T, size_t N, typename D>
template<size_t M>
std::enable_if_t<(M > 1), size_t> TensorBase<T, N, D>::height() const {
    return _shape.y;
}

template<typename T, size_t N, typename D>
template<size_t M>
std::enable_if_t<(M > 2), size_t> TensorBase<T, N, D>::depth() const {
    return _shape.z;
}

template<typename T, size_t N, typename D>
bool TensorBase<T, N, D>::isEmpty() const {
    return length() == 0;
}

template<typename T, size_t N, typename D>
size_t TensorBase<T, N, D>::length() const {
    return product<size_t, N>(_shape, 1);
}

template<typename T, size_t N, typename D>
typename TensorBase<T, N, D>::iterator TensorBase<T, N, D>::begin() {
    return _ptr;
}

template<typename T, size_t N, typename D>
typename TensorBase<T, N, D>::const_iterator TensorBase<T, N, D>::begin() const {
    return _ptr;
}

template<typename T, size_t N, typename D>
typename TensorBase<T, N, D>::iterator TensorBase<T, N, D>::end() {
    return _ptr + length();
}

template<typename T, size_t N, typename D>
typename TensorBase<T, N, D>::const_iterator TensorBase<T, N, D>::end() const {
    return _ptr + length();
}

template<typename T, size_t N, typename D>
typename TensorBase<T, N, D>::iterator TensorBase<T, N, D>::rbegin() {
    return end() - 1;
}

template<typename T, size_t N, typename D>
typename TensorBase<T, N, D>::const_iterator TensorBase<T, N, D>::rbegin() const {
    return end() - 1;
}

template<typename T, size_t N, typename D>
typename TensorBase<T, N, D>::iterator TensorBase<T, N, D>::rend() {
    return begin() - 1;
}

template<typename T, size_t N, typename D>
typename TensorBase<T, N, D>::const_iterator TensorBase<T, N, D>::rend() const {
    return begin() - 1;
}

template<typename T, size_t N, typename D>
T &TensorBase<T, N, D>::at(size_t i) {
    return _ptr[i];
}

template<typename T, size_t N, typename D>
const T &TensorBase<T, N, D>::at(size_t i) const {
    return _ptr[i];
}

template<typename T, size_t N, typename D>
template<typename... Args>
T &TensorBase<T, N, D>::at(size_t i, Args... args) {
    return data()[index(i, args...)];
}

template<typename T, size_t N, typename D>
template<typename... Args>
const T &TensorBase<T, N, D>::at(size_t i, Args... args) const {
    return _ptr[index(i, args...)];
}

template<typename T, size_t N, typename D>
T &TensorBase<T, N, D>::at(const Vector<size_t, N> &idx) {
    return data()[index(idx)];
}

template<typename T, size_t N, typename D>
const T &TensorBase<T, N, D>::at(const Vector<size_t, N> &idx) const {
    return data()[index(idx)];
}

template<typename T, size_t N, typename D>
T &TensorBase<T, N, D>::operator[](size_t i) {
    return at(i);
}

template<typename T, size_t N, typename D>
const T &TensorBase<T, N, D>::operator[](size_t i) const {
    return at(i);
}

template<typename T, size_t N, typename D>
template<typename... Args>
T &TensorBase<T, N, D>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template<typename T, size_t N, typename D>
template<typename... Args>
const T &TensorBase<T, N, D>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template<typename T, size_t N, typename D>
T &TensorBase<T, N, D>::operator()(const Vector<size_t, N> &idx) {
    return at(idx);
}

template<typename T, size_t N, typename D>
const T &TensorBase<T, N, D>::operator()(const Vector<size_t, N> &idx) const {
    return at(idx);
}

template<typename T, size_t N, typename D>
TensorBase<T, N, D>::TensorBase() : _shape{} {}

template<typename T, size_t N, typename D>
TensorBase<T, N, D>::TensorBase(const TensorBase &other) {
    setPtrAndSize(other._ptr, other._shape);
}

template<typename T, size_t N, typename D>
TensorBase<T, N, D>::TensorBase(TensorBase &&other) {
    *this = std::move(other);
}

template<typename T, size_t N, typename D>
template<typename... Args>
void TensorBase<T, N, D>::setPtrAndSize(T *ptr, size_t ni, Args... args) {
    setPtrAndSize(ptr, Vector<size_t, N>{ni, args...});
}

template<typename T, size_t N, typename D>
void TensorBase<T, N, D>::setPtrAndSize(T *data, Vector<size_t, N> size) {
    _ptr = data;
    _shape = size;
}

template<typename T, size_t N, typename D>
void TensorBase<T, N, D>::clearPtrAndSize() {
    setPtrAndSize(nullptr, Vector<size_t, N>{});
}

template<typename T, size_t N, typename D>
void TensorBase<T, N, D>::swapPtrAndSize(TensorBase &other) {
    std::swap(_ptr, other._ptr);
    std::swap(_shape, other._shape);
}

template<typename T, size_t N, typename D>
TensorBase<T, N, D> &TensorBase<T, N, D>::operator=(const TensorBase &other) {
    setPtrAndSize(other.data(), other.shape());
    return *this;
}

template<typename T, size_t N, typename D>
TensorBase<T, N, D> &TensorBase<T, N, D>::operator=(TensorBase &&other) {
    setPtrAndSize(other.data(), other.shape());
    other.setPtrAndSize(nullptr, Vector<size_t, N>{});
    return *this;
}

template<typename T, size_t N, typename D>
template<typename... Args>
size_t TensorBase<T, N, D>::_index(size_t d, size_t i, Args... args) const {
    return i + _shape[d] * _index(d + 1, args...);
}

template<typename T, size_t N, typename D>
size_t TensorBase<T, N, D>::_index(size_t, size_t i) const {
    return i;
}

template<typename T, size_t N, typename D>
template<size_t... I>
size_t TensorBase<T, N, D>::_index(const Vector<size_t, N> &idx,
                                   std::index_sequence<I...>) const {
    return index(idx[I]...);
}

// MARK: Tensor

// CTOR
template<typename T, size_t N>
Tensor<T, N>::Tensor() : Base() {}

template<typename T, size_t N>
Tensor<T, N>::Tensor(const Vector<size_t, N> &size_, const T &initVal) : Tensor() {
    _data.resize(product<size_t, N>(size_, 1), initVal);
    Base::setPtrAndSize(_data.data(), size_);
}

template<typename T, size_t N>
template<typename... Args>
Tensor<T, N>::Tensor(size_t nx, Args... args) {
    Vector<size_t, N> size_;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(size_, initVal, nx, args...);
    _data.resize(product<size_t, N>(size_, 1), initVal);
    Base::setPtrAndSize(_data.data(), size_);
}

template<typename T, size_t N>
Tensor<T, N>::Tensor(NestedInitializerListsT<T, N> lst) {
    Vector<size_t, N> newSize{};
    internal::GetSizeFromInitList<T, N, N>::call(newSize, lst);
    _data.resize(product<size_t, N>(newSize, 1));
    Base::setPtrAndSize(_data.data(), newSize);
    internal::SetTensorFromInitList<T, N, N>::call(*this, lst);
}

template<typename T, size_t N>
template<typename OtherDerived>
Tensor<T, N>::Tensor(const TensorBase<T, N, OtherDerived> &other) : Tensor() {
    copyFrom(other);
}

template<typename T, size_t N>
template<typename OtherDerived>
Tensor<T, N>::Tensor(const TensorBase<const T, N, OtherDerived> &other) : Tensor() {
    copyFrom(other);
}

template<typename T, size_t N>
Tensor<T, N>::Tensor(const Tensor &other) : Tensor() {
    copyFrom(other);
}

template<typename T, size_t N>
Tensor<T, N>::Tensor(Tensor &&other) : Tensor() {
    *this = std::move(other);
}

template<typename T, size_t N>
template<typename D>
void Tensor<T, N>::copyFrom(const TensorBase<T, N, D> &other) {
    resize(other.shape());
    forEachIndex(Vector<size_t, N>{}, other.shape(),
                 [&](auto... idx) { this->at(idx...) = other(idx...); });
}

template<typename T, size_t N>
template<typename D>
void Tensor<T, N>::copyFrom(const TensorBase<const T, N, D> &other) {
    resize(other.shape());
    forEachIndex(Vector<size_t, N>{}, other.shape(),
                 [&](auto... idx) { this->at(idx...) = other(idx...); });
}

template<typename T, size_t N>
void Tensor<T, N>::fill(const T &val) {
    std::fill(_data.begin(), _data.end(), val);
}

template<typename T, size_t N>
void Tensor<T, N>::resize(Vector<size_t, N> size_, const T &initVal) {
    Tensor newTensor(size_, initVal);
    Vector<size_t, N> minSize = min(_shape, newTensor._shape);
    forEachIndex(minSize,
                 [&](auto... idx) { newTensor(idx...) = (*this)(idx...); });
    *this = std::move(newTensor);
}

template<typename T, size_t N>
template<typename... Args>
void Tensor<T, N>::resize(size_t nx, Args... args) {
    Vector<size_t, N> size_;
    T initVal;
    internal::GetSizeAndInitVal<T, N, N - 1>::call(size_, initVal, nx, args...);

    resize(size_, initVal);
}

template<typename T, size_t N>
template<size_t M>
std::enable_if_t<(M == 1), void> Tensor<T, N>::append(const T &val) {
    _data.push_back(val);
    Base::setPtrAndSize(_data.data(), _data.size());
}

template<typename T, size_t N>
template<typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> Tensor<T, N>::append(
    const TensorBase<T, N, OtherDerived> &extra) {
    _data.insert(_data.end(), extra.begin(), extra.end());
    Base::setPtrAndSize(_data.data(), _data.size());
}

template<typename T, size_t N>
template<typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> Tensor<T, N>::append(
    const TensorBase<const T, N, OtherDerived> &extra) {
    _data.insert(_data.end(), extra.begin(), extra.end());
    Base::setPtrAndSize(_data.data(), _data.size());
}

template<typename T, size_t N>
void Tensor<T, N>::clear() {
    Base::clearPtrAndSize();
    _data.clear();
}

template<typename T, size_t N>
void Tensor<T, N>::swap(Tensor &other) {
    Base::swapPtrAndSize(other);
    std::swap(_data, other._data);
}

template<typename T, size_t N>
TensorView<T, N> Tensor<T, N>::view() {
    return TensorView<T, N>(*this);
};

template<typename T, size_t N>
TensorView<const T, N> Tensor<T, N>::view() const {
    return TensorView<const T, N>(*this);
};

template<typename T, size_t N>
template<typename OtherDerived>
Tensor<T, N> &Tensor<T, N>::operator=(
    const TensorBase<T, N, OtherDerived> &other) {
    copyFrom(other);
    return *this;
}

template<typename T, size_t N>
template<typename OtherDerived>
Tensor<T, N> &Tensor<T, N>::operator=(
    const TensorBase<const T, N, OtherDerived> &other) {
    copyFrom(other);
    return *this;
}

template<typename T, size_t N>
Tensor<T, N> &Tensor<T, N>::operator=(const Tensor &other) {
    copyFrom(other);
    return *this;
}

template<typename T, size_t N>
Tensor<T, N> &Tensor<T, N>::operator=(Tensor &&other) {
    _data = std::move(other._data);
    Base::setPtrAndSize(other.data(), other.shape());
    other.setPtrAndSize(nullptr, Vector<size_t, N>{});

    return *this;
}

}// namespace vox