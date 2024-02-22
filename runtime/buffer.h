//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>
#include "device.h"

namespace vox {
template<typename T>
class Buffer {
public:
    explicit Buffer(const Device &device) : _device{device} {
        ContextGuard guard(device.primary_context());
        check_cuda(cudaMalloc(&_device_buffer, sizeof(T)));
    }

    ~Buffer() {
        ContextGuard guard(_device.primary_context());
        check_cuda(cudaFree(_device_buffer));
    }

private:
    const Device &_device;
    std::vector<T> _host_buffer;
    void *_device_buffer{};
};

template<typename T>
Buffer<T> create_buffer(uint32_t index = 0) {
    const auto &d = device(index);
    return Buffer<T>{d};
}

}// namespace vox