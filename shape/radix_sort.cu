//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "radix_sort.h"
#include <cub/cub.cuh>

namespace vox {
RadixSort::RadixSort(uint32_t index) : index{index} {
}

void RadixSort::reserve(int n) {
    cub::DoubleBuffer<int> d_keys;
    cub::DoubleBuffer<int> d_values;

    // compute temporary memory required
    size_t sort_temp_size;
    check_cuda(cub::DeviceRadixSort::SortPairs(
        nullptr,
        sort_temp_size,
        d_keys,
        d_values,
        n, 0, 32,
        stream(index).handle()));

    if (sort_temp_size > mem.size()) {
        mem.resizeUninitialized(sort_temp_size);
    }
}

void RadixSort::execute(int *keys, int *values, int n) {
    cub::DoubleBuffer<int> d_keys(keys, keys + n);
    cub::DoubleBuffer<int> d_values(values, values + n);

    reserve(n);

    // sort
    auto size = mem.size();
    check_cuda(cub::DeviceRadixSort::SortPairs(
        (void *)mem.data(),
        size,
        d_keys,
        d_values,
        n, 0, 32,
        stream(index).handle()));

    if (d_keys.Current() != keys)
        mem.cudaCopyDeviceToDevice((uint8_t *)d_keys.Current(), sizeof(int) * n, (uint8_t *)keys);

    if (d_values.Current() != values)
        mem.cudaCopyDeviceToDevice((uint8_t *)d_values.Current(), sizeof(int) * n, (uint8_t *)values);
}

}// namespace vox