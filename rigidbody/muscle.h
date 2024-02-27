//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "tensor/tensor.h"
#include "math/matrix.h"

namespace vox {
class MuscleModelBuilder {
public:
    struct MuscleParams {
        /// Force scaling
        float f0;
        /// Muscle length
        float lm;
        /// Tendon length
        float lt;
        /// Maximally efficient muscle length
        float lmax;
        float pen;
    };
    Tensor1<size_t> muscle_start;
    Tensor1<MuscleParams> muscle_params;
    Tensor1<float> muscle_activation;
    Tensor1<int> muscle_bodies;
    Tensor1<Vector3F> muscle_points;

    [[nodiscard]] size_t muscle_count() const;

    /// Adds a muscle-tendon activation unit.
    /// \param bodies A list of body indices for each waypoint
    /// \param positions A list of positions of each waypoint in the body's local frame
    /// \param f0 Force scaling
    /// \param lm Muscle length
    /// \param lt Tendon length
    /// \param lmax Maximally efficient muscle length
    /// \param pen
    /// \return The index of the muscle in the model
    size_t add_muscle(std::initializer_list<int> bodies, std::initializer_list<Vector3F> positions,
                      float f0, float lm, float lt, float lmax, float pen);
};
}// namespace vox