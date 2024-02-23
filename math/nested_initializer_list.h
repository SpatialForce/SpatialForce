//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <initializer_list>

namespace vox {

template<typename T, size_t N>
struct NestedInitializerLists {
    using type =
        std::initializer_list<typename NestedInitializerLists<T, N - 1>::type>;
};

template<typename T>
struct NestedInitializerLists<T, 0> {
    using type = T;
};

//- Aliases.

template<typename T, size_t N>
using NestedInitializerListsT = typename NestedInitializerLists<T, N>::type;

}// namespace vox