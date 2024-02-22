//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "device.h"

namespace vox {
template<typename T>
class Buffer {
public:
    Buffer(const Device &device, size_t n) : _device{device}, _n{n} {
        ContextGuard guard(device.primary_context());
        _byte_size = sizeof(T) * n;
        check_cuda(cudaMalloc(&_handle, _byte_size));
    }

    ~Buffer() {
        ContextGuard guard(_device.primary_context());
        check_cuda(cudaFree(_handle));
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
    size_t _n;
    size_t _byte_size;
    const Device &_device;
    T *_handle{};
};

template<typename T>
Buffer<T> create_buffer(size_t n, uint32_t index = 0) {
    const auto &d = device(index);
    return {d, n};
}

template<typename T>
void sync_h2d(void *src, Buffer<T> &dst) {
    auto &device = dst.device();
    ContextGuard guard(device.primary_context());
    check_cuda(cudaMemcpyAsync(dst.handle(), src, dst.byte_size(), cudaMemcpyHostToDevice, device.stream.handle()));
}

template<typename T>
void sync_d2h(Buffer<T> &src, void *dst) {
    auto &device = src.device();
    ContextGuard guard(device.primary_context());
    check_cuda(cudaMemcpyAsync(dst, src.handle(), src.byte_size(), cudaMemcpyDeviceToHost, device.stream.handle()));
}

template<typename T>
void sync_d2d(const Buffer<T> &src, Buffer<T> &dst) {
    auto &device = src.device();
    ContextGuard guard(device.primary_context());
    check_cuda(cudaMemcpyAsync(dst.handle(), src.handle(), src.byte_size(), cudaMemcpyDeviceToDevice, device.stream.handle()));
}

template<typename T>
void memset(Buffer<T> &dst, int value) {
    auto &device = dst.device();
    ContextGuard guard(device.primary_context());
    check_cuda(cudaMemsetAsync(dst, value, dst.size(), device.stream.handle()));
}

}// namespace vox