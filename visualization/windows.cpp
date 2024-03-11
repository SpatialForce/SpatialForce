//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "windows.h"
#include "renderer.h"
#include "interactor.h"

namespace vox {
Windows::Windows(std::string_view title, int width, int height) {
    _handle->SetSize(width, height);
    _handle->SetWindowName(title.data());
    _handle->AddRenderer(Renderer::instance().handle());
    Interactor::instance().handle()->SetRenderWindow(_handle);
}

void Windows::render() {
    _handle->Render();
    Interactor::instance().handle()->Start();
}

}// namespace vox