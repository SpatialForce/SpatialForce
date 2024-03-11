//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <vtkRenderWindowInteractor.h>

namespace vox {
class Interactor {
public:
    static Interactor &instance() {
        static Interactor instance;
        return instance;
    }

    inline vtkNew<vtkRenderWindowInteractor> &handle() {
        return _handle;
    }

private:
    vtkNew<vtkRenderWindowInteractor> _handle;
};
}// namespace vox