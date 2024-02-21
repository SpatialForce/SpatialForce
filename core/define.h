//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstdlib>
#include <cstdint>

namespace vox {
#if !defined(__CUDACC__)
#define CUDA_CALLABLE
#define CUDA_CALLABLE_DEVICE
#else
#define CUDA_CALLABLE __host__ __device__
#define CUDA_CALLABLE_DEVICE __device__
#endif
}// namespace vox