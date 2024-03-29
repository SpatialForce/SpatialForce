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

template<typename T, size_t Rows, size_t Cols>
struct GetScalarType<Matrix<T, Rows, Cols>> {
    using value = T;
};

}// namespace vox