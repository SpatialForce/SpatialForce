//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vtkRenderWindow.h>

namespace vox {
class Windows {
public:
    Windows(std::string_view title, int width, int height);

    void render();

private:
    vtkNew<vtkRenderWindow> _handle;
};
}// namespace vox