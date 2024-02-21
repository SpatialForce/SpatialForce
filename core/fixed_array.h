//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "define.h"

namespace vox {
template<typename Type, size_t Length>
struct fixed_array_t {
    Type storage[Length];

    CUDA_CALLABLE Type &operator[](unsigned i) { return storage[i]; }

    CUDA_CALLABLE const Type &operator[](unsigned i) const { return storage[i]; }
};

}// namespace vox