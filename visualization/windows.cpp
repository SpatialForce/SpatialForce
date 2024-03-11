//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "windows.h"
#include "renderer.h"

namespace vox {
Windows::Windows(std::string_view title, int width, int height) {
    _handle->SetSize(width, height);
    _handle->SetWindowName(title.data());
}

void Windows::render() {
    _handle->Render();
    Renderer::instance().bindWindow(this);
}

}// namespace vox