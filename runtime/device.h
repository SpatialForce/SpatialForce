//  Copyright (c) 2024 Feng Yang
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

    int ptx_version{};
    cudaDeviceProp props{};
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
private:
    DeviceInfo *_info{};

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
    Device(Device &&device) noexcept = default;
};

/// initialize the CUDA runtime.
void init();

void deinit();

/// Returns the number of CUDA devices supported in this environment.
size_t device_count();

/// Returns the device identified by the argument.
/// \param index device index
/// \return Device
const Device &device(uint32_t index = 0);

/// Manually synchronize the calling CPU thread with any outstanding CUDA work on the specified device
//
//    This method allows the host application code to ensure that any kernel launches
//    or memory copies have completed.
/// \param index Device to synchronize
void synchronize(uint32_t index = 0);

}// namespace vox