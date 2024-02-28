//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "runtime/cuda_buffer.h"

namespace vox {
class RadixSort {
public:
    explicit RadixSort(uint32_t index = 0);

    void reserve(int n);

    void execute(int *keys, int *values, int n);

private:
    uint32_t index;
    CudaBuffer<uint8_t> mem;
};
}// namespace vox