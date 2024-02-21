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
    struct DeviceInfo {
        static constexpr int kNameLen = 128;

        CUdevice device = -1;
        int ordinal = -1;
        char name[kNameLen] = "";
        int arch = 0;
        int is_uva = 0;
        int is_memory_pool_supported = 0;
    };

    // cached info for all devices, indexed by ordinal
    std::vector<DeviceInfo> all_devices;

    Device();
};

const Device::DeviceInfo &device_info(uint32_t index);

}// namespace vox