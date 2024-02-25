//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {

#ifdef __CUDACC__

namespace internal {

template<typename T, size_t N, size_t I>
struct CudaBlockCopyHelper {
    template<typename... RemainingIndices>
    __host__ __device__ static void call(CudaTensorView<const T, N> src,
                                         CudaStdArray<size_t, N> size,
                                         CudaTensorView<T, N> dst,
                                         RemainingIndices... indices) {
        for (size_t i = 0; i < size[I - 1]; ++i) {
            CudaBlockCopyHelper<T, N, I - 1>::call(src, size, dst, i,
                                                   indices...);
        }
    }
};

template<typename T, size_t N>
struct CudaBlockCopyHelper<T, N, 1> {
    template<typename... RemainingIndices>
    __host__ __device__ static void call(CudaTensorView<const T, N> src,
                                         CudaStdArray<size_t, N> size,
                                         CudaTensorView<T, N> dst,
                                         RemainingIndices... indices) {
        for (size_t i = 0; i < size[0]; ++i) {
            dst(i, indices...) = src(i, indices...);
        }
    }
};

template<typename T, size_t N>
__global__ void cudaBlockCopyKernelN(CudaTensorView<const T, N> src,
                                     CudaStdArray<size_t, N> size,
                                     CudaTensorView<T, N> dst) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size[N - 1]) {
        CudaBlockCopyHelper<T, N, N - 1>::call(src, size, dst, i);
    }
}

template<typename T>
__global__ void cudaBlockCopyKernel1(CudaTensorView<const T, 1> src,
                                     CudaStdArray<size_t, 1> size,
                                     CudaTensorView<T, 1> dst) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size[0]) {
        dst[i] = src[i];
    }
}

template<typename T, size_t N>
struct CudaBlockCopy {
    static void call(CudaTensorView<const T, N> src,
                     CudaStdArray<size_t, N> size, CudaTensorView<T, N> dst) {
        if (size[N - 1] == 0) {
            return;
        }

        // Assuming i-major
        unsigned int numBlocks, numThreads;
        cudaComputeGridSize((unsigned int)size[N - 1], 256, numBlocks,
                            numThreads);
        cudaBlockCopyKernelN<<<numBlocks, numThreads>>>(src, size, dst);
        CUDA_CHECK_LAST_ERROR("Failed executing cudaBlockCopyKernelN");
    }
};

template<typename T>
struct CudaBlockCopy<T, 1> {
    static void call(CudaTensorView<const T, 1> src,
                     CudaStdArray<size_t, 1> size, CudaTensorView<T, 1> dst) {
        if (size[0] == 0) {
            return;
        }

        // Assuming i-major
        unsigned int numBlocks, numThreads;
        cudaComputeGridSize((unsigned int)size[0], 256, numBlocks, numThreads);
        cudaBlockCopyKernel1<<<numBlocks, numThreads>>>(src, size, dst);
        CUDA_CHECK_LAST_ERROR("Failed executing cudaBlockCopyKernel1");
    }
};

}// namespace internal

#endif// __CUDACC__

////////////////////////////////////////////////////////////////////////////////
// MARK: CudaTensor

template<typename T, size_t N>
CudaTensor<T, N>::CudaTensor() : Base() {}

template<typename T, size_t N>
CudaTensor<T, N>::CudaTensor(const CudaStdArray<size_t, N> &shape_,
                             const T &initVal)
    : CudaTensor() {
    // TODO: Replace CudaStdArray with Vector
    size_t l = shape_[0];
    for (size_t i = 1; i < N; ++i) {
        l *= shape_[i];
    }
    _data.resize(l, initVal);
    Base::setPtrAndShape(_data.data(), shape_);
}

#ifdef __CUDACC__

template<typename T, size_t N>
template<typename... Args>
CudaTensor<T, N>::CudaTensor(size_t nx, Args... args) : CudaTensor() {
    // TODO: Replace CudaStdArray with Vector
    Vector<size_t, N> newSizeV;
    T initVal;
    internal::GetShapeAndInitVal<T, N, N - 1>::call(newSizeV, initVal, nx,
                                                    args...);
    CudaStdArray<size_t, N> newSize(newSizeV);
    CudaTensor newArray(newSize, initVal);
    *this = std::move(newArray);
}

template<typename T, size_t N>
CudaTensor<T, N>::CudaTensor(NestedInitializerListsT<T, N> lst) : CudaTensor() {
    Vector<size_t, N> newSize;
    internal::GetShapeFromInitList<T, N, N>::call(newSize, lst);

    Tensor<T, N> newCpuArray(newSize);
    internal::SetTensorFromInitList<T, N, N>::call(newCpuArray, lst);
    copyFrom(newCpuArray);
}

#endif// __CUDACC__

template<typename T, size_t N>
template<size_t M>
CudaTensor<T, N>::CudaTensor(
    const std::enable_if_t<(M == 1), std::vector<T>> &vec)
    : CudaTensor() {
    copyFrom(vec);
}

template<typename T, size_t N>
template<typename OtherDerived>
CudaTensor<T, N>::CudaTensor(const TensorBase<T, N, OtherDerived> &other)
    : CudaTensor() {
    copyFrom(other);
}

template<typename T, size_t N>
template<typename OtherDerived>
CudaTensor<T, N>::CudaTensor(const CudaTensorBase<T, N, OtherDerived> &other)
    : CudaTensor() {
    copyFrom(other);
}

template<typename T, size_t N>
CudaTensor<T, N>::CudaTensor(const CudaTensor &other) : CudaTensor() {
    copyFrom(other);
}

template<typename T, size_t N>
CudaTensor<T, N>::CudaTensor(CudaTensor &&other) noexcept : CudaTensor() {
    *this = std::move(other);
}

template<typename T, size_t N>
template<typename A, size_t M>
std::enable_if_t<(M == 1), void> CudaTensor<T, N>::copyFrom(const std::vector<T, A> &vec) {
    CudaTensor newArray(vec.size());
    newArray._data.copyFrom(vec);
    newArray.setPtrAndShape(newArray._data.data(), newArray.shape());
    *this = std::move(newArray);
}

template<typename T, size_t N>
template<typename OtherDerived>
void CudaTensor<T, N>::copyFrom(const TensorBase<T, N, OtherDerived> &other) {
    CudaTensor newArray(CudaStdArray<size_t, N>(other.shape()));
    _data.cudaCopyHostToDevice(other.data(), other.length(), newArray.data());
    *this = std::move(newArray);
}

template<typename T, size_t N>
template<typename OtherDerived>
void CudaTensor<T, N>::copyFrom(const TensorBase<const T, N, OtherDerived> &other) {
    CudaTensor newArray(CudaStdArray<size_t, N>(other.shape()));
    _data.cudaCopyHostToDevice(other.data(), other.length(), newArray.data());
    *this = std::move(newArray);
}

template<typename T, size_t N>
template<typename OtherDerived>
void CudaTensor<T, N>::copyFrom(const CudaTensorBase<T, N, OtherDerived> &other) {
    CudaTensor newArray(other.shape());
    _data.cudaCopyDeviceToDevice(other.data(), other.length(), newArray.data());
    *this = std::move(newArray);
}

template<typename T, size_t N>
template<typename OtherDerived>
void CudaTensor<T, N>::copyFrom(const CudaTensorBase<const T, N, OtherDerived> &other) {
    CudaTensor newArray(other.shape());
    _data.cudaCopyDeviceToDevice(other.data(), other.length(), newArray.data());
    *this = std::move(newArray);
}

template<typename T, size_t N>
template<typename A, size_t M>
std::enable_if_t<(M == 1), void> CudaTensor<T, N>::copyTo(std::vector<T, A> &vec) {
    vec.resize(length());
    _data.cudaCopyDeviceToHost(data(), length(), vec.data());
}

template<typename T, size_t N>
void CudaTensor<T, N>::copyTo(Tensor<T, N> &other) {
    other.resize(_shape.toVector());
    _data.cudaCopyDeviceToHost(data(), length(), other.data());
}

template<typename T, size_t N>
void CudaTensor<T, N>::copyTo(TensorView<T, N> &other) {
    ASSERT(_shape.toVector() == other.shape());
    _data.cudaCopyDeviceToHost(data(), length(), other.data());
}

template<typename T, size_t N>
void CudaTensor<T, N>::copyTo(CudaTensor<T, N> &other) {
    other.resize(CudaStdArray<size_t, N>(_shape.toVector()));
    _data.cudaCopyDeviceToDevice(data(), length(), other.data());
}

template<typename T, size_t N>
void CudaTensor<T, N>::copyTo(CudaTensorView<T, N> &other) {
    ASSERT(length() == other.length());
    _data.cudaCopyDeviceToDevice(data(), length(), other.data());
}

template<typename T, size_t N>
void CudaTensor<T, N>::fill(const T &val) {
    _data.fill(val);
}

#ifdef __CUDACC__

template<typename T, size_t N>
void CudaTensor<T, N>::resize(CudaStdArray<size_t, N> newShape,
                              const T &initVal) {
    // TODO: Replace with Vector
    CudaTensor newArray(newShape, initVal);
    CudaStdArray<size_t, N> minSize;
    for (size_t i = 0; i < N; ++i) {
        minSize[i] = std::min(_shape[i], newArray._shape[i]);
    }

    internal::CudaBlockCopy<T, N>::call(view(), minSize, newArray.view());

    *this = std::move(newArray);
}

template<typename T, size_t N>
template<typename... Args>
void CudaTensor<T, N>::resize(size_t nx, Args... args) {
    // TODO: Replace CudaStdArray with Vector
    Vector<size_t, N> newSizeV;
    T initVal;
    internal::GetShapeAndInitVal<T, N, N - 1>::call(newSizeV, initVal, nx,
                                                    args...);

    CudaStdArray<size_t, N> newSize(newSizeV);
    resize(newSize, initVal);
}

#endif// __CUDACC__

template<typename T, size_t N>
template<size_t M>
std::enable_if_t<(M == 1), void> CudaTensor<T, N>::append(const T &val) {
    _data.push_back(val);
    Base::setPtrAndShape(_data.data(), _data.size());
}

template<typename T, size_t N>
template<typename A, size_t M>
std::enable_if_t<(M == 1), void> CudaTensor<T, N>::append(
    const std::vector<T, A> &extra) {
    _data.append(extra);
    _shape[0] = _data.size();
}

template<typename T, size_t N>
template<typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> CudaTensor<T, N>::append(const TensorBase<T, N, OtherDerived> &extra) {
    CudaTensor newArray(length() + extra.length());
    _data.cudaCopy(data(), length(), newArray.data());
    _data.cudaCopyHostToDevice(extra.data(), extra.length(), newArray.data() + _shape[0]);
    swap(newArray);
}

template<typename T, size_t N>
template<typename OtherDerived, size_t M>
std::enable_if_t<(M == 1), void> CudaTensor<T, N>::append(
    const CudaTensorBase<T, N, OtherDerived> &extra) {
    CudaTensor newArray(length() + extra.length());
    _data.cudaCopy(data(), length(), newArray.data());
    _data.cudaCopy(extra.data(), extra.length(), newArray.data() + _shape[0]);
    swap(newArray);
}

template<typename T, size_t N>
void CudaTensor<T, N>::clear() {
    Base::clearPtrAndShape();
    _data.clear();
}

template<typename T, size_t N>
void CudaTensor<T, N>::swap(CudaTensor &other) {
    Base::swapPtrAndShape(other);
    _data.swap(other._data);
}

template<typename T, size_t N>
CudaTensorView<T, N> CudaTensor<T, N>::view() {
    return CudaTensorView<T, N>(*this);
}

template<typename T, size_t N>
CudaTensorView<const T, N> CudaTensor<T, N>::view() const {
    return CudaTensorView<const T, N>(*this);
}

template<typename T, size_t N>
template<size_t M>
CudaTensor<T, N> &CudaTensor<T, N>::operator=(
    const std::enable_if_t<(M == 1), std::vector<T>> &vec) {
    copyFrom(vec);
    return *this;
}

template<typename T, size_t N>
template<typename OtherDerived>
CudaTensor<T, N> &CudaTensor<T, N>::operator=(const TensorBase<T, N, OtherDerived> &other) {
    copyFrom(other);
    return *this;
}

template<typename T, size_t N>
template<typename OtherDerived>
CudaTensor<T, N> &CudaTensor<T, N>::operator=(const TensorBase<const T, N, OtherDerived> &other) {
    copyFrom(other);
    return *this;
}

template<typename T, size_t N>
template<typename OtherDerived>
CudaTensor<T, N> &CudaTensor<T, N>::operator=(const CudaTensorBase<T, N, OtherDerived> &other) {
    copyFrom(other);
    return *this;
}

template<typename T, size_t N>
template<typename OtherDerived>
CudaTensor<T, N> &CudaTensor<T, N>::operator=(const CudaTensorBase<const T, N, OtherDerived> &other) {
    copyFrom(other);
    return *this;
}

template<typename T, size_t N>
CudaTensor<T, N> &CudaTensor<T, N>::operator=(const CudaTensor &other) {
    _data = other._data;
    Base::setPtrAndShape(_data.data(), other.shape());
    return *this;
}

template<typename T, size_t N>
CudaTensor<T, N> &CudaTensor<T, N>::operator=(CudaTensor &&other) noexcept {
    swap(other);
    other.clear();
    return *this;
}

template<typename T, size_t N>
typename CudaTensor<T, N>::host_reference
CudaTensor<T, N>::at(size_t i) {
    return host_reference(_ptr + i, _data.device());
}

template<typename T, size_t N>
CudaTensor<T, N>::Base::value_type CudaTensor<T, N>::at(size_t i) const {
    return (T)host_reference(_ptr + i, _data.device());
}

template<typename T, size_t N>
template<typename... Args>
typename CudaTensor<T, N>::host_reference
CudaTensor<T, N>::at(size_t i, Args... args) {
    return at(index(i, args...));
}

template<typename T, size_t N>
template<typename... Args>
CudaTensor<T, N>::Base::value_type CudaTensor<T, N>::at(size_t i, Args... args) const {
    return at(index(i, args...));
}

template<typename T, size_t N>
typename CudaTensor<T, N>::host_reference
CudaTensor<T, N>::at(const CudaStdArray<size_t, N> &idx) {
    return at(index(idx));
}

template<typename T, size_t N>
CudaTensor<T, N>::Base::value_type CudaTensor<T, N>::at(const CudaStdArray<size_t, N> &idx) const {
    return at(index(idx));
}

template<typename T, size_t N>
typename CudaTensor<T, N>::host_reference
CudaTensor<T, N>::operator[](size_t i) {
    return at(i);
}

template<typename T, size_t N>
CudaTensor<T, N>::Base::value_type CudaTensor<T, N>::operator[](size_t i) const {
    return at(i);
}

template<typename T, size_t N>
template<typename... Args>
typename CudaTensor<T, N>::host_reference
CudaTensor<T, N>::operator()(size_t i, Args... args) {
    return at(i, args...);
}

template<typename T, size_t N>
template<typename... Args>
CudaTensor<T, N>::Base::value_type CudaTensor<T, N>::operator()(size_t i, Args... args) const {
    return at(i, args...);
}

template<typename T, size_t N>
typename CudaTensor<T, N>::host_reference
CudaTensor<T, N>::operator()(const CudaStdArray<size_t, N> &idx) {
    return at(idx);
}

template<typename T, size_t N>
CudaTensor<T, N>::Base::value_type CudaTensor<T, N>::operator()(const CudaStdArray<size_t, N> &idx) const {
    return at(idx);
}

}// namespace vox
