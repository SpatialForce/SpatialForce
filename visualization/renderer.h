//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vtkRenderer.h>

namespace vox {
class Renderer {
public:
    static Renderer &instance();

    Renderer();

    vtkNew<vtkRenderer> &handle();

    void resetCamera();

    void addActor(vtkProp *actor);

private:
    vtkNew<vtkRenderer> _handle;
};
}// namespace vox