//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "math/matrix.h"
#include "math/nested_initializer_list.h"

#include <algorithm>
#include <functional>
#include <vector>

namespace vox {

// MARK: TensorBase

template<typename T, size_t N, typename DerivedTensor>
class TensorBase {
public:
    using Derived = DerivedTensor;
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = T *;
    using const_pointer = const T *;
    using iterator = T *;
    using const_iterator = const T *;

    virtual ~TensorBase() = default;

    size_t index(size_t i) const;

    template<typename... Args>
    size_t index(size_t i, Args... args) const;

    template<size_t... I>
    size_t index(const Vector<size_t, N> &idx) const;

    pointer data();

    const_pointer data() const;

    const Vector<size_t, N> &size() const;

    template<size_t M = N>
    std::enable_if_t<(M > 0), size_t> width() const;

    template<size_t M = N>
    std::enable_if_t<(M > 1), size_t> height() const;

    template<size_t M = N>
    std::enable_if_t<(M > 2), size_t> depth() const;

    bool isEmpty() const;

    size_t length() const;

    iterator begin();

    const_iterator begin() const;

    iterator end();

    const_iterator end() const;

    iterator rbegin();

    const_iterator rbegin() const;

    iterator rend();

    const_iterator rend() const;

    reference at(size_t i);

    const_reference at(size_t i) const;

    template<typename... Args>
    reference at(size_t i, Args... args);

    template<typename... Args>
    const_reference at(size_t i, Args... args) const;

    reference at(const Vector<size_t, N> &idx);

    const_reference at(const Vector<size_t, N> &idx) const;

    reference operator[](size_t i);

    const_reference operator[](size_t i) const;

    template<typename... Args>
    reference operator()(size_t i, Args... args);

    template<typename... Args>
    const_reference operator()(size_t i, Args... args) const;

    reference operator()(const Vector<size_t, N> &idx);

    const_reference operator()(const Vector<size_t, N> &idx) const;

protected:
    pointer _ptr = nullptr;
    Vector<size_t, N> _size;

    TensorBase();

    TensorBase(const TensorBase &other);

    TensorBase(TensorBase &&other);

    template<typename... Args>
    void setPtrAndSize(pointer ptr, size_t ni, Args... args);

    void setPtrAndSize(pointer data, Vector<size_t, N> size);

    void swapPtrAndSize(TensorBase &other);

    void clearPtrAndSize();

    TensorBase &operator=(const TensorBase &other);

    TensorBase &operator=(TensorBase &&other);

private:
    template<typename... Args>
    size_t _index(size_t d, size_t i, Args... args) const;

    size_t _index(size_t, size_t i) const;

    template<size_t... I>
    size_t _index(const Vector<size_t, N> &idx,
                  std::index_sequence<I...>) const;
};

// MARK: Tensor

template<typename T, size_t N>
class TensorView;

template<typename T, size_t N>
class Tensor final : public TensorBase<T, N, Tensor<T, N>> {
    using Base = TensorBase<T, N, Tensor<T, N>>;
    using Base::_size;
    using Base::at;
    using Base::clearPtrAndSize;
    using Base::setPtrAndSize;
    using Base::swapPtrAndSize;

public:
    // CTOR
    Tensor();

    Tensor(const Vector<size_t, N> &size_, const T &initVal = T{});

    template<typename... Args>
    Tensor(size_t nx, Args... args);

    Tensor(NestedInitializerListsT<T, N> lst);

    template<typename OtherDerived>
    Tensor(const TensorBase<T, N, OtherDerived> &other);

    template<typename OtherDerived>
    Tensor(const TensorBase<const T, N, OtherDerived> &other);

    Tensor(const Tensor &other);

    Tensor(Tensor &&other);

    template<typename D>
    void copyFrom(const TensorBase<T, N, D> &other);

    template<typename D>
    void copyFrom(const TensorBase<const T, N, D> &other);

    void fill(const T &val);

    // resize
    void resize(Vector<size_t, N> size_, const T &initVal = T{});

    template<typename... Args>
    void resize(size_t nx, Args... args);

    template<size_t M = N>
    std::enable_if_t<(M == 1), void> append(const T &val);

    template<typename OtherDerived, size_t M = N>
    std::enable_if_t<(M == 1), void> append(
        const TensorBase<T, N, OtherDerived> &extra);

    template<typename OtherDerived, size_t M = N>
    std::enable_if_t<(M == 1), void> append(
        const TensorBase<const T, N, OtherDerived> &extra);

    void clear();

    void swap(Tensor &other);

    // Views
    TensorView<T, N> view();

    TensorView<const T, N> view() const;

    // Assignment Operators
    template<typename OtherDerived>
    Tensor &operator=(const TensorBase<T, N, OtherDerived> &other);

    template<typename OtherDerived>
    Tensor &operator=(const TensorBase<const T, N, OtherDerived> &other);

    Tensor &operator=(const Tensor &other);

    Tensor &operator=(Tensor &&other);

private:
    std::vector<T> _data;
};

template<class T>
using Tensor1 = Tensor<T, 1>;

template<class T>
using Tensor2 = Tensor<T, 2>;

template<class T>
using Tensor3 = Tensor<T, 3>;

template<class T>
using Tensor4 = Tensor<T, 4>;

}// namespace vox

#include "tensor-inl.h"