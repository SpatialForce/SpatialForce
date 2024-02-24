//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>

#include <cstdio>

namespace vox {
#define check_cuda(code) (check_cuda_result(code, __FILE__, __LINE__))
#define check_cu(code) (check_cu_result(code, __FILE__, __LINE__))

bool check_cuda_result(cudaError_t code, const char *file, int line);

bool check_cu_result(CUresult result, const char *file, int line);

#define _CUDA_CHECK(result, msg, file, line)                                \
    if (result != cudaSuccess) {                                            \
        fprintf(stderr, "CUDA error at %s:%d code=%d (%s) \"%s\" \n", file, \
                line, static_cast<unsigned int>(result),                    \
                cudaGetErrorString(result), msg);                           \
        cudaDeviceReset();                                                  \
        exit(EXIT_FAILURE);                                                 \
    }

#define CUDA_CHECK(expression) \
    _CUDA_CHECK((expression), #expression, __FILE__, __LINE__)

#define CUDA_CHECK_LAST_ERROR(msg) \
    _CUDA_CHECK(cudaGetLastError(), msg, __FILE__, __LINE__)

inline unsigned int cudaDivRoundUp(unsigned int a,
                                   unsigned int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

inline void cudaComputeGridSize(unsigned int n,
                                unsigned int blockSize,
                                unsigned int &numBlocks,
                                unsigned int &numThreads) {
    numThreads = std::min(blockSize, n);
    numBlocks = cudaDivRoundUp(n, numThreads);
}

//
// Scoped CUDA context guard
//
// Behaviour on entry
// - If the given `context` is NULL, do nothing.
// - If the given `context` is the same as the current context, do nothing.
// - If the given `context` is different from the current context, make the given context current.
//
// Behaviour on exit
// - If the current context did not change on entry, do nothing.
// - If the `restore` flag was true on entry, make the previous context current.
//
// Default exit behaviour policy
// - If the `restore` flag is omitted on entry, fall back on the global `always_restore` flag.
// - This allows us to easily change the default behaviour of the guards.
//
class ContextGuard {
public:
    // default policy for restoring contexts
    static bool always_restore;

    explicit ContextGuard(CUcontext context, bool restore = always_restore)
        : need_restore(false) {
        if (context) {
            if (check_cu(cuCtxGetCurrent(&prev_context)) && context != prev_context)
                need_restore = check_cu(cuCtxSetCurrent(context)) && restore;
        }
    }

    explicit ContextGuard(void *context, bool restore = always_restore)
        : ContextGuard(static_cast<CUcontext>(context), restore) {
    }

    ~ContextGuard() {
        if (need_restore)
            check_cu(cuCtxSetCurrent(prev_context));
    }

private:
    CUcontext prev_context{};
    bool need_restore;
};
constexpr void *CURRENT_CONTEXT = nullptr;

}// namespace vox