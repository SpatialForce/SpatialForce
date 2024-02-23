//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {

//! Returns the type of the value itself.
template<typename T>
struct GetScalarType {
    typedef T value;
};

}// namespace vox