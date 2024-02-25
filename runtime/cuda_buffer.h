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
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

    class Reference {
    public:
        Reference(pointer p, const Device &device)
            : _ptr(p), _device{device} {}

        Reference(const Reference &other)
            : _ptr(other._ptr), _device{other._device} {}

        Reference &operator=(const value_type &val) {
            ContextGuard guard(_device.primary_context());
            check_cuda(cudaMemcpy(_ptr, &val, sizeof(value_type), cudaMemcpyHostToDevice));
            return *this;
        }

        operator value_type() const {
            std::remove_const_t<value_type> tmp{};
            ContextGuard guard(_device.primary_context());
            check_cuda(cudaMemcpy(&tmp, _ptr, sizeof(value_type), cudaMemcpyDeviceToHost));
            return tmp;
        }
    private:
        const Device &_device;
        pointer _ptr;
    };

    explicit CudaBuffer(const Device &device = vox::device(0));

    explicit CudaBuffer(size_t n, const value_type &initVal = value_type{},
                        const Device &device = vox::device(0));

    template<typename A>
    explicit CudaBuffer(const std::vector<T, A> &other,
                        const Device &device = vox::device(0));

    CudaBuffer(const CudaBuffer &other);

    CudaBuffer(CudaBuffer &&other) noexcept;

    ~CudaBuffer();

    pointer data();

    const_pointer data() const;

    [[nodiscard]] size_t size() const;

    Reference at(size_t i);

    T at(size_t i) const;

    void clear();

    void fill(const value_type &val);

    void resize(size_t n, const value_type &initVal = value_type{});

    void resizeUninitialized(size_t n);

    void swap(CudaBuffer &other);

    void push_back(const value_type &val);

    void append(const value_type &val);

    void append(const CudaBuffer &other);

    void cudaCopy(const T *src, size_t n, T *dst,
                  cudaMemcpyKind kind = cudaMemcpyDeviceToDevice) {
        ContextGuard guard(_device.primary_context());
        CUDA_CHECK(cudaMemcpy(dst, src, n * sizeof(T), kind));
    }

    void cudaCopyDeviceToDevice(const T *src, size_t n, T *dst) {
        cudaCopy(src, n, dst, cudaMemcpyDeviceToDevice);
    }

    void cudaCopyHostToDevice(const T *src, size_t n, T *dst) {
        cudaCopy(src, n, dst, cudaMemcpyHostToDevice);
    }

    void cudaCopyDeviceToHost(const T *src, size_t n, T *dst) {
        cudaCopy(src, n, dst, cudaMemcpyDeviceToHost);
    }

    void copyFrom(const CudaBuffer &other);

    template<typename A>
    void copyFrom(const std::vector<T, A> &other);

    template<typename A>
    void copyTo(std::vector<T, A> &other);

    template<typename A>
    CudaBuffer &operator=(const std::vector<T, A> &other);

    CudaBuffer &operator=(const CudaBuffer &other);

    CudaBuffer &operator=(CudaBuffer &&other) noexcept;

    Reference operator[](size_t i);

    T operator[](size_t i) const;

    [[nodiscard]] inline const Device &device() const;

private:
    const Device &_device;
    size_t _size{};
    pointer _ptr{nullptr};
};

//-----------------------------------------------------------------------------------------------------------------------------------
template<typename T>
struct HostDeviceVector {
    CudaBuffer<T> device_buffer;
    std::vector<T> host_buffer;

    explicit HostDeviceVector(uint32_t index = 0)
        : device_buffer{index} {}

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
        device_buffer.resize(n);
    }

    void sync_d2h() {
        device_buffer.copyTo(host_buffer);
    }

    void sync_h2d() {
        device_buffer = host_buffer;
    }

    auto begin() {
        return host_buffer.begin();
    }

    auto end() {
        return host_buffer.end();
    }

    array_t<T> view() {
        return {device_buffer.data(), (int)device_buffer.size()};
    }
};

}// namespace vox

#include "cuda_buffer-inl.h"