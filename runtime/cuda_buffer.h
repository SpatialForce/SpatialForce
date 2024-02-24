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
#ifdef __CUDACC__
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
#endif// __CUDACC__

template<typename T>
class CudaBuffer {
public:
    using value_type = T;
    using reference = T &;
    using const_reference = const T &;
    using pointer = value_type *;
    using const_pointer = const value_type *;

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

    void clear();

    void fill(const value_type &val);

    void resize(size_t n, const value_type &initVal = value_type{});

    void resizeUninitialized(size_t n);

    void swap(CudaBuffer &other);

    template<typename A>
    void copyFrom(const std::vector<T, A> &other);

    void copyFrom(const CudaBuffer &other);

    template<typename A>
    void copyTo(std::vector<T, A> &other);

    template<typename A>
    CudaBuffer &operator=(const std::vector<T, A> &other);

    CudaBuffer &operator=(const CudaBuffer &other);

    CudaBuffer &operator=(CudaBuffer &&other) noexcept;

    [[nodiscard]] inline size_t byte_size() const;

    [[nodiscard]] inline const Device &device() const;

private:
    void _alloc(size_t size) {
        _size = size;
        ContextGuard guard(_device.primary_context());
        _byte_size = sizeof(T) * size;
        check_cuda(cudaMalloc(&_handle, _byte_size));
    }

    size_t _size{};
    size_t _byte_size{};
    const Device &_device;
    pointer _handle{nullptr};
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