//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>
#include <string_view>
#include "cuda_util.h"
#include "stream.h"

namespace vox {
struct DeviceInfo {
    static constexpr int kNameLen = 128;

    CUdevice handle = -1;
    CUuuid uuid = {0};
    int ordinal = -1;
    int pci_domain_id = -1;
    int pci_bus_id = -1;
    int pci_device_id = -1;
    char name[kNameLen] = "";
    int arch = 0;
    int is_uva = 0;
    int is_memory_pool_supported = 0;

    CUcontext primary_context{};

public:
    void primary_context_retain();
};

class Device {
public:
    // asynchronous work
    Stream stream;
    // CUDA default stream for some synchronous operations
    Stream null_stream;

    [[nodiscard]] const DeviceInfo &info() const;

    [[nodiscard]] CUcontext primary_context() const;

    [[nodiscard]] std::string_view name() const {
        return _info->name;
    }

    [[nodiscard]] int arch() const {
        return _info->arch;
    }

    explicit Device(DeviceInfo *info);

private:
    DeviceInfo *_info;
};

void init();

size_t device_count();

const Device &device(uint32_t index);

}// namespace vox