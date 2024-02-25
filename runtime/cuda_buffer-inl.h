//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {
template<typename T>
__global__ void cudaFillKernel(T *dst, size_t n, T val) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = val;
    }
}

template<typename T>
void cudaFill(T *dst, size_t n, const T &val) {
    if (n == 0) {
        return;
    }

    unsigned int numBlocks, numThreads;
    cudaComputeGridSize((unsigned int)n, 256, numBlocks, numThreads);
    cudaFillKernel<<<numBlocks, numThreads>>>(dst, n, val);
    CUDA_CHECK_LAST_ERROR("Failed executing cudaFillKernel");
}

template<typename T>
CudaBuffer<T>::CudaBuffer(const Device &device) : _device{device} {}

template<typename T>
CudaBuffer<T>::CudaBuffer(size_t n, const value_type &initVal,
                          const Device &device) : _device{device} {
    resizeUninitialized(n);
    cudaFill(_ptr, n, initVal);
}

template<typename T>
template<typename A>
CudaBuffer<T>::CudaBuffer(const std::vector<T, A> &other,
                          const Device &device)
    : CudaBuffer(other.size(), value_type{}, device) {
    cudaCopyHostToDevice(other.data(), _size, _ptr);
}

template<typename T>
CudaBuffer<T>::CudaBuffer(const CudaBuffer &other)
    : CudaBuffer(other.size(), value_type{}, other.device()) {
    cudaCopyDeviceToDevice(other._ptr, _size, _ptr);
}

template<typename T>
CudaBuffer<T>::CudaBuffer(CudaBuffer &&other) noexcept
    : _device{other.device()} {
    *this = std::move(other);
}

template<typename T>
CudaBuffer<T>::~CudaBuffer() {
    clear();
}

template<typename T>
CudaBuffer<T>::pointer CudaBuffer<T>::data() {
    return _ptr;
}

template<typename T>
CudaBuffer<T>::const_pointer CudaBuffer<T>::data() const {
    return _ptr;
}

template<typename T>
size_t CudaBuffer<T>::size() const {
    return _size;
}

template<typename T>
typename CudaBuffer<T>::Reference CudaBuffer<T>::at(size_t i) {
    Reference r(_ptr + i);
    return r;
}

template<typename T>
T CudaBuffer<T>::at(size_t i) const {
    T tmp;
    cudaCopyDeviceToHost(_ptr + i, 1, &tmp);
    return tmp;
}

template<typename T>
void CudaBuffer<T>::clear() {
    if (_ptr != nullptr) {
        ContextGuard guard(_device.primary_context());
        check_cuda(cudaFree(_ptr));
    }
    _ptr = nullptr;
    _size = 0;
}

template<typename T>
void CudaBuffer<T>::fill(const value_type &val) {
    cudaFill(_ptr, _size, val);
}

template<typename T>
void CudaBuffer<T>::resize(size_t n, const value_type &initVal) {
    CudaBuffer newBuffer(n, initVal, device());
    cudaCopy(_ptr, std::min(n, _size), newBuffer._ptr);
    swap(newBuffer);
}

template<typename T>
void CudaBuffer<T>::resizeUninitialized(size_t n) {
    clear();
    _size = n;
    ContextGuard guard(_device.primary_context());
    check_cuda(cudaMalloc(&_ptr, sizeof(T) * n));
}

template<typename T>
void CudaBuffer<T>::swap(CudaBuffer &other) {
    std::swap(_ptr, other._ptr);
    std::swap(_size, other._size);
}

template<typename T>
void CudaBuffer<T>::push_back(const value_type &val) {
    CudaBuffer newBuffer;
    newBuffer.resizeUninitialized(_size + 1);
    cudaCopy(_ptr, _size, newBuffer._ptr);
    cudaCopyHostToDevice(&val, 1, newBuffer._ptr + _size);
    swap(newBuffer);
}

template<typename T>
void CudaBuffer<T>::append(const value_type &val) {
    push_back(val);
}

template<typename T>
void CudaBuffer<T>::append(const CudaBuffer &other) {
    CudaBuffer newBuffer;
    newBuffer.resizeUninitialized(_size + other._size);
    cudaCopy(_ptr, _size, newBuffer._ptr);
    cudaCopy(other._ptr, other._size, newBuffer._ptr + _size);
    swap(newBuffer);
}

template<typename T>
void CudaBuffer<T>::copyFrom(const CudaBuffer &other) {
    if (_size == other.size()) {
        cudaCopyDeviceToDevice(other.data(), _size, _ptr);
    } else {
        CudaBuffer newBuffer(other);
        swap(newBuffer);
    }
}

template<typename T>
template<typename A>
void CudaBuffer<T>::copyFrom(const std::vector<T, A> &other) {
    if (_size == other.size()) {
        cudaCopyHostToDevice(other.data(), _size, _ptr);
    } else {
        CudaBuffer newBuffer(other);
        swap(newBuffer);
    }
}

template<typename T>
template<typename A>
void CudaBuffer<T>::copyTo(std::vector<T, A> &other) {
    other.resize(_size);
    cudaCopyDeviceToHost(_ptr, _size, other.data());
}

template<typename T>
template<typename A>
CudaBuffer<T> &CudaBuffer<T>::operator=(const std::vector<T, A> &other) {
    copyFrom(other);
    return *this;
}

template<typename T>
CudaBuffer<T> &CudaBuffer<T>::operator=(const CudaBuffer &other) {
    copyFrom(other);
    return *this;
}

template<typename T>
CudaBuffer<T> &CudaBuffer<T>::operator=(CudaBuffer &&other) noexcept {
    clear();
    swap(other);
    return *this;
}

template<typename T>
typename CudaBuffer<T>::Reference CudaBuffer<T>::operator[](size_t i) {
    return at(i);
}

template<typename T>
T CudaBuffer<T>::operator[](size_t i) const {
    return at(i);
}

template<typename T>
inline const Device &CudaBuffer<T>::device() const {
    return _device;
}
}// namespace vox