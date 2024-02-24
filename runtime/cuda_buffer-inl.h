//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {
template<typename T>
CudaBuffer<T>::CudaBuffer(const Device &device) : _device{device} {}

template<typename T>
CudaBuffer<T>::CudaBuffer(size_t n, const value_type &initVal,
                          const Device &device) : _device{device} {
    resizeUninitialized(n);
    cudaFill(_handle, n, initVal);
}

template<typename T>
template<typename A>
CudaBuffer<T>::CudaBuffer(const std::vector<T, A> &other,
                          const Device &device)
    : CudaBuffer(other.size(), value_type{}, device) {
    ContextGuard guard(device.primary_context());
    check_cuda(cudaMemcpyAsync(_handle, other.data(), byte_size(), cudaMemcpyHostToDevice, device.stream.handle()));
}

template<typename T>
CudaBuffer<T>::CudaBuffer(const CudaBuffer &other)
    : CudaBuffer(other.size(), value_type{}, other.device()) {
    ContextGuard guard(_device.primary_context());
    check_cuda(cudaMemcpyAsync(_handle, other.data(), byte_size(), cudaMemcpyHostToDevice, _device.stream.handle()));
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
    return _handle;
}

template<typename T>
CudaBuffer<T>::const_pointer CudaBuffer<T>::data() const {
    return _handle;
}

template<typename T>
size_t CudaBuffer<T>::size() const {
    return _size;
}

template<typename T>
void CudaBuffer<T>::clear() {
    if (_handle != nullptr) {
        ContextGuard guard(_device.primary_context());
        check_cuda(cudaFree(_handle));
    }
    _handle = nullptr;
    _size = 0;
    _byte_size = 0;
}

template<typename T>
void CudaBuffer<T>::fill(const value_type &val) {
    cudaFill(_handle, _size, val);
}

template<typename T>
void CudaBuffer<T>::resize(size_t n, const value_type &initVal) {
    CudaBuffer newBuffer(n, initVal, device());
    ContextGuard guard(_device.primary_context());
    check_cuda(cudaMemcpyAsync(newBuffer._handle, _handle, std::min(n, _size) * sizeof(T), cudaMemcpyDeviceToDevice, _device.stream.handle()));
    swap(newBuffer);
}

template<typename T>
void CudaBuffer<T>::resizeUninitialized(size_t n) {
    if (_handle) {
        clear();
    }
    _alloc(n);
}

template<typename T>
void CudaBuffer<T>::swap(CudaBuffer &other) {
    std::swap(_handle, other._handle);
    std::swap(_size, other._size);
    std::swap(_byte_size, other._byte_size);
}

template<typename T>
template<typename A>
void CudaBuffer<T>::copyFrom(const std::vector<T, A> &other) {
    if (_size == other.size()) {
        ContextGuard guard(_device.primary_context());
        check_cuda(cudaMemcpyAsync(_handle, other.data(), byte_size(), cudaMemcpyHostToDevice, _device.stream.handle()));
    } else {
        CudaBuffer newBuffer(other, device());
        swap(newBuffer);
    }
}

template<typename T>
void CudaBuffer<T>::copyFrom(const CudaBuffer &other) {
    if (_size == other.size()) {
        ContextGuard guard(_device.primary_context());
        check_cuda(cudaMemcpyAsync(other.data(), _handle, byte_size(), cudaMemcpyDeviceToDevice, _device.stream.handle()));
    } else {
        CudaBuffer newBuffer(other);
        swap(newBuffer);
    }
}

template<typename T>
template<typename A>
void CudaBuffer<T>::copyTo(std::vector<T, A> &other) {
    other.resize(_size);
    ContextGuard guard(_device.primary_context());
    check_cuda(cudaMemcpyAsync(other.data(), _handle, byte_size(), cudaMemcpyDeviceToHost, _device.stream.handle()));
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
inline size_t CudaBuffer<T>::byte_size() const {
    return _byte_size;
}

template<typename T>
inline const Device &CudaBuffer<T>::device() const {
    return _device;
}
}// namespace vox