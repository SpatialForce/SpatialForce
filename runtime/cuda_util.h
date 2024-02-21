//  Copyright (c) 2023 Feng Yang
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

}// namespace vox