//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <memory>

namespace vox::fields {

template<typename TYPE, int order>
class PolyInfo {
};

template<typename TYPE, int order>
using PolyInfoPtr = std::shared_ptr<PolyInfo<TYPE, order>>;

}// namespace vox::fields