//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "muscle.h"

namespace vox {
size_t MuscleModelBuilder::muscle_count() const {
    return muscle_start.width();
}

size_t MuscleModelBuilder::add_muscle(std::initializer_list<int> bodies, std::initializer_list<Vector3F> positions,
                                      float f0, float lm, float lt, float lmax, float pen) {
    auto n = bodies.size();

    muscle_start.append(muscle_bodies.width());
    muscle_params.append({f0, lm, lt, lmax, pen});
    muscle_activation.append(0.0);

    for (const auto &body : bodies) {
        muscle_bodies.append(body);
    }
    for (const auto &position : positions) {
        muscle_points.append(position);
    }

    // return the index of the muscle
    return muscle_start.width() - 1;
}
}// namespace vox