//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "device.h"
#include "core/array.h"
#include <vector>
#include <array>

namespace vox {
template<typename T>
class CudaBuffer {
public:
    explicit CudaBuffer(const Device &device) : _device{device} {}

    ~CudaBuffer() {
        _free();
    }

    void alloc(size_t n) {
        if (_handle) {
            _free();
        }
        _alloc(n);
    }

    inline T *handle() {
        return _handle;
    }

    [[nodiscard]] inline size_t byte_size() const {
        return _byte_size;
    }

    [[nodiscard]] inline size_t size() const {
        return _n;
    }

    [[nodiscard]] inline const Device &device() const {
        return _device;
    }

private:
    void _alloc(size_t n) {
        _n = n;
        ContextGuard guard(_device.primary_context());
        _byte_size = sizeof(T) * n;
        check_cuda(cudaMalloc(&_handle, _byte_size));
    }

    void _free() {
        ContextGuard guard(_device.primary_context());
        check_cuda(cudaFree(_handle));
    }

    size_t _n{};
    size_t _byte_size{};
    const Device &_device;
    T *_handle{nullptr};
};

template<typename T>
CudaBuffer<T> create_buffer(uint32_t index = 0) {
    const auto &d = device(index);
    return CudaBuffer<T>{d};
}

template<typename T>
void sync_h2d(void *src, CudaBuffer<T> &dst) {
    auto &device = dst.device();
    ContextGuard guard(device.primary_context());
    check_cuda(cudaMemcpyAsync(dst.handle(), src, dst.byte_size(), cudaMemcpyHostToDevice, device.stream.handle()));
}

template<typename T>
void sync_d2h(CudaBuffer<T> &src, void *dst) {
    auto &device = src.device();
    ContextGuard guard(device.primary_context());
    check_cuda(cudaMemcpyAsync(dst, src.handle(), src.byte_size(), cudaMemcpyDeviceToHost, device.stream.handle()));
}

template<typename T>
void sync_d2d(const CudaBuffer<T> &src, CudaBuffer<T> &dst) {
    auto &device = src.device();
    ContextGuard guard(device.primary_context());
    check_cuda(cudaMemcpyAsync(dst.handle(), src.handle(), src.byte_size(), cudaMemcpyDeviceToDevice, device.stream.handle()));
}

template<typename T>
void memset(CudaBuffer<T> &dst, int value) {
    auto &device = dst.device();
    ContextGuard guard(device.primary_context());
    check_cuda(cudaMemsetAsync(dst, value, dst.size(), device.stream.handle()));
}

//-----------------------------------------------------------------------------------------------------------------------------------
template<typename T>
struct HostDeviceVector {
    CudaBuffer<T> device_buffer;
    std::vector<T> host_buffer;

    explicit HostDeviceVector(uint32_t index = 0)
        : device_buffer{create_buffer<T>(index)} {}

    HostDeviceVector<T> &operator=(const std::vector<T> &host) {
        host_buffer = host;
        return *this;
    }

    HostDeviceVector<T> &operator=(std::vector<T> &&host) {
        host_buffer = std::move(host);
        return *this;
    }

    T &operator[](size_t i) {
        return host_buffer[i];
    }

    const T &operator[](size_t i) const {
        return host_buffer[i];
    }

    void resize(size_t n) {
        host_buffer.resize(n);
        device_buffer.alloc(n);
    }

    void sync_d2h() {
        if (device_buffer.size() != host_buffer.size()) {
            host_buffer.resize(device_buffer.size());
        }
        vox::sync_d2h(device_buffer, host_buffer.data());
    }

    void sync_h2d() {
        if (device_buffer.size() != host_buffer.size()) {
            device_buffer.alloc(host_buffer.size());
        }
        vox::sync_h2d(host_buffer.data(), device_buffer);
    }

    auto begin() {
        return host_buffer.begin();
    }

    auto end() {
        return host_buffer.end();
    }

    array_t<T> view() {
        return {device_buffer.handle(), (int)device_buffer.size()};
    }
};

//-----------------------------------------------------------------------------------------------------------------------------------
template<typename T, size_t N>
struct HostDeviceArray {
    CudaBuffer<T> device_buffer;
    std::array<T, N> host_buffer;

    explicit HostDeviceArray(uint32_t index = 0)
        : device_buffer{create_buffer<T>(index)} {
        device_buffer.alloc(N);
    }

    HostDeviceArray<T, N> &operator=(const std::array<T, N> &host) {
        host_buffer = host;
        return *this;
    }

    HostDeviceArray<T, N> &operator=(std::array<T, N> &&host) {
        host_buffer = std::move(host);
        return *this;
    }

    void sync_d2h() {
        vox::sync_d2h(device_buffer, host_buffer.data());
    }

    void sync_h2d() {
        vox::sync_h2d(host_buffer.data(), device_buffer);
    }

    auto begin() {
        return host_buffer.begin();
    }

    auto end() {
        return host_buffer.end();
    }

    array_t<T> view() {
        return {device_buffer.handle(), (int)device_buffer.size()};
    }
};

}// namespace vox