//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "tensor/tensor.h"
#include "cuda_tensor_view.h"
#include "cuda_buffer.h"

namespace vox {
template<typename T, size_t N>
class CudaTensor final : public CudaTensorBase<T, N, CudaTensor<T, N>> {
    using Base = CudaTensorBase<T, N, CudaTensor<T, N>>;
    using Base::_shape;
    using Base::setPtrAndShape;
    using Base::swapPtrAndShape;

public:
    using Base::at;
    using Base::clearPtrAndShape;
    using Base::data;
    using Base::length;

    // CTOR
    CudaTensor();

    explicit CudaTensor(const CudaStdArray<size_t, N> &shape_, const T &initVal = T{});

    template<typename... Args>
    explicit CudaTensor(size_t nx, Args... args);

    explicit CudaTensor(NestedInitializerListsT<T, N> lst);

    template<size_t M = N>
    explicit CudaTensor(const std::enable_if_t<(M == 1), std::vector<T>> &vec);

    template<typename OtherDerived>
    explicit CudaTensor(const TensorBase<T, N, OtherDerived> &other);

    template<typename OtherDerived>
    explicit CudaTensor(const CudaTensorBase<T, N, OtherDerived> &other);

    CudaTensor(const CudaTensor &other);

    CudaTensor(CudaTensor &&other) noexcept;

    template<typename A, size_t M = N>
    std::enable_if_t<(M == 1), void> copyFrom(const std::vector<T, A> &vec);

    template<typename OtherDerived>
    void copyFrom(const TensorBase<T, N, OtherDerived> &other);

    template<typename OtherDerived>
    void copyFrom(const TensorBase<const T, N, OtherDerived> &other);

    template<typename OtherDerived>
    void copyFrom(const CudaTensorBase<T, N, OtherDerived> &other);

    template<typename OtherDerived>
    void copyFrom(const CudaTensorBase<const T, N, OtherDerived> &other);

    template<typename A, size_t M = N>
    std::enable_if_t<(M == 1), void> copyTo(std::vector<T, A> &vec);

    void copyTo(Tensor<T, N> &other);

    void copyTo(TensorView<T, N> &other);

    void copyTo(CudaTensor<T, N> &other);

    void copyTo(CudaTensorView<T, N> &other);

    void fill(const T &val);

    // resize
    void resize(CudaStdArray<size_t, N> newShape, const T &initVal = T{});

    template<typename... Args>
    void resize(size_t nx, Args... args);

//    template<size_t M = N>
//    std::enable_if_t<(M == 1), void> append(const T &val);
//
//    template<typename A, size_t M = N>
//    std::enable_if_t<(M == 1), void> append(const std::vector<T, A> &extra);
//
//    template<typename OtherDerived, size_t M = N>
//    std::enable_if_t<(M == 1), void> append(const TensorBase<T, N, OtherDerived> &extra);
//
//    template<typename OtherDerived, size_t M = N>
//    std::enable_if_t<(M == 1), void> append(const CudaTensorBase<T, N, OtherDerived> &extra);

    void clear();

    void swap(CudaTensor &other);

    // Views
    CudaTensorView<T, N> view();

    CudaTensorView<const T, N> view() const;

    // Assignment Operators
    template<size_t M = N>
    CudaTensor &operator=(const std::enable_if_t<(M == 1), std::vector<T>> &vec);

    template<typename OtherDerived>
    CudaTensor &operator=(const TensorBase<T, N, OtherDerived> &other);

    template<typename OtherDerived>
    CudaTensor &operator=(const TensorBase<const T, N, OtherDerived> &other);

    template<typename OtherDerived>
    CudaTensor &operator=(const CudaTensorBase<T, N, OtherDerived> &other);

    template<typename OtherDerived>
    CudaTensor &operator=(const CudaTensorBase<const T, N, OtherDerived> &other);

    CudaTensor &operator=(const CudaTensor &other);

    CudaTensor &operator=(CudaTensor &&other) noexcept;

private:
    CudaBuffer<T> _data;
};

template<class T>
using CudaTensor1 = CudaTensor<T, 1>;

template<class T>
using CudaTensor2 = CudaTensor<T, 2>;

template<class T>
using CudaTensor3 = CudaTensor<T, 3>;

template<class T>
using CudaTensor4 = CudaTensor<T, 4>;

}// namespace vox

#include "cuda_tensor-inl.h"
