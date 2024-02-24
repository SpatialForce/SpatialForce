//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cstdlib>
#include <cstdint>
#include <cassert>
#include <cstdio>
#include <cmath>

#if !defined(__CUDACC__)
#define CUDA_CALLABLE
#define CUDA_CALLABLE_DEVICE
#else
#define CUDA_CALLABLE __host__ __device__
#define CUDA_CALLABLE_DEVICE __device__
#endif

static constexpr double kEps = 1e-10;

// MARK: Debug mode
#if defined(NDEBUG)
#define ASSERT(x)
#else
#include <cassert>
#define ASSERT(x) assert(x)
#endif

// MARK: C++ exceptions
#ifdef __cplusplus
#include <stdexcept>
#define THROW_INVALID_ARG_IF(expression)          \
    if (expression) {                             \
        throw std::invalid_argument(#expression); \
    }
#define THROW_INVALID_ARG_WITH_MESSAGE_IF(expression, message) \
    if (expression) {                                          \
        throw std::invalid_argument(message);                  \
    }
#endif
