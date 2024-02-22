//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vector>
#include "cuda_util.h"

namespace vox {
class Device {
public:
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

private:
    friend class DeviceInfo;
    void _primary_context_retain();
};

void init();

size_t device_count();

const Device &device(uint32_t index);

}// namespace vox