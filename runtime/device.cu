//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "device.h"
#include <cassert>

namespace vox {
class DeviceInfo {
public:
    // cached info for all devices, indexed by ordinal
    std::vector<Device> all_devices;

    DeviceInfo();
};

DeviceInfo::DeviceInfo() {
    if (!check_cu(cuInit(0))) return;

    int deviceCount = 0;
    if (check_cu(cuDeviceGetCount(&deviceCount))) {
        all_devices.resize(deviceCount);

        for (int i = 0; i < deviceCount; i++) {
            CUdevice device;
            if (check_cu(cuDeviceGet(&device, i))) {
                // query device info
                all_devices[i].device = device;
                all_devices[i].ordinal = i;
                check_cu(cuDeviceGetName(all_devices[i].name, Device::kNameLen, device));
                check_cu(cuDeviceGetAttribute(&all_devices[i].is_uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
                check_cu(cuDeviceGetAttribute(&all_devices[i].is_memory_pool_supported,
                                              CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device));
                int major = 0;
                int minor = 0;
                check_cu(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
                check_cu(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
                all_devices[i].arch = 10 * major + minor;
            }
        }
    }
}

size_t device_count() {
    int deviceCount = 0;
    cuDeviceGetCount(&deviceCount);
    return deviceCount;
}

const Device &device(uint32_t index) {
    static DeviceInfo cuda_device;
    assert(index < cuda_device.all_devices.size());
    return cuda_device.all_devices[index];
}

}// namespace vox