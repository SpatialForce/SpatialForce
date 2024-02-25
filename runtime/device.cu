//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "device.h"
#include <map>

namespace vox {
// Dummy kernel for retrieving PTX version.
template<int dummy_arg>
__global__ void dummy_k() {}

void DeviceInfo::primary_context_retain() {
    check_cu(cuDevicePrimaryCtxRetain(&primary_context, handle));
}

static std::vector<DeviceInfo> all_devices;

void init() {
    cuInit(0);

    int deviceCount = 0;
    if (check_cu(cuDeviceGetCount(&deviceCount))) {
        all_devices.resize(deviceCount);

        for (int i = 0; i < deviceCount; i++) {
            CUdevice device;
            if (check_cu(cuDeviceGet(&device, i))) {
                cudaGetDeviceProperties(&all_devices[i].props, i);
                cudaFuncAttributes attr{};
                check_cuda(cudaFuncGetAttributes(&attr, dummy_k<0>));
                all_devices[i].ptx_version = attr.ptxVersion;

                // query device info
                all_devices[i].handle = device;
                all_devices[i].ordinal = i;
                check_cu(cuDeviceGetName(all_devices[i].name, DeviceInfo::kNameLen, device));
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
                all_devices[i].primary_context_retain();
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
    static std::map<CUdevice, Device> cuda_device;

    auto handle = all_devices[index].handle;
    auto it = cuda_device.find(handle);
    if (it != cuda_device.end()) {
        return it->second;
    } else {
        auto result = cuda_device.emplace(handle, Device(&all_devices[index]));
        return result.first->second;
    }
}

void synchronize(uint32_t index) {
    auto &d = device(index);
    ContextGuard guard(d.primary_context());
    check_cu(cuCtxSynchronize());
}

//---------------------------------------------------------------------------------------------------------------
Device::Device(DeviceInfo *info)
    : _info{info},
      null_stream(*this, nullptr),
      stream(*this) {
}

const DeviceInfo &Device::info() const {
    return *_info;
}

CUcontext Device::primary_context() const {
    return _info->primary_context;
}

}// namespace vox