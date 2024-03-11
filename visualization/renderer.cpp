//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "renderer.h"
#include "windows.h"
#include <vtkNamedColors.h>

namespace vox {
Renderer &Renderer::instance() {
    static Renderer instance;
    return instance;
}

Renderer::Renderer() {
    vtkNew<vtkNamedColors> colors;
    // Set the background color.
    std::array<unsigned char, 4> bkg{{26, 51, 102, 255}};
    colors->SetColor("BkgColor", bkg.data());
    _handle->SetBackground(colors->GetColor3d("BkgColor").GetData());
}

vtkNew<vtkRenderer> &Renderer::handle() {
    return _handle;
}

void Renderer::bindWindow(Windows* win) {
    _interactor->SetRenderWindow(win->handle());
    win->handle()->AddRenderer(_handle);

    vtkNew<vtkCameraOrientationWidget> orientationWidget;
    orientationWidget->SetParentRenderer(_handle);
    orientationWidget->On();

    _interactor->Start();
}

void Renderer::resetCamera() {
    _handle->ResetCamera();
}

void Renderer::addActor(vtkProp *actor) {
    _handle->AddActor(actor);
}

}// namespace vox