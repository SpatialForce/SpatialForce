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
    int deviceCount = 0;
    if (check_cu(cuDeviceGetCount(&deviceCount))) {
        all_devices.resize(deviceCount);

        for (int i = 0; i < deviceCount; i++) {
            CUdevice device;
            if (check_cu(cuDeviceGet(&device, i))) {
                // query device info
                all_devices[i].handle = device;
                all_devices[i].ordinal = i;
                check_cu(cuDeviceGetName(all_devices[i].name, Device::kNameLen, device));
                check_cu(cuDeviceGetUuid(&all_devices[i].uuid, device));
                check_cu(cuDeviceGetAttribute(&all_devices[i].pci_domain_id, CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID, device));
                check_cu(cuDeviceGetAttribute(&all_devices[i].pci_bus_id, CU_DEVICE_ATTRIBUTE_PCI_BUS_ID, device));
                check_cu(cuDeviceGetAttribute(&all_devices[i].pci_device_id, CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID, device));
                check_cu(cuDeviceGetAttribute(&all_devices[i].is_uva, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, device));
                check_cu(cuDeviceGetAttribute(&all_devices[i].is_memory_pool_supported, CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED, device));
                int major = 0;
                int minor = 0;
                check_cu(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
                check_cu(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
                all_devices[i].arch = 10 * major + minor;

                all_devices[i]._primary_context_retain();
            }
        }
    }
}

void init() {
    cuInit(0);
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

void Device::_primary_context_retain() {
    check_cu(cuDevicePrimaryCtxRetain(&primary_context, handle));
}

}// namespace vox