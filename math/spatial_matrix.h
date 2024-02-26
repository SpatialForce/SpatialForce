//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "spatial_vector.h"

namespace vox {
template<typename Type>
using SpatialMatrix = Matrix<Type, 6, 6>;

using SpatialMatrixF = SpatialMatrix<float>;
using SpatialMatrixD = SpatialMatrix<double>;

template<typename Type>
inline CUDA_CALLABLE SpatialMatrix<Type> spatial_adjoint(const Matrix<Type, 3, 3> &R, const Matrix<Type, 3, 3> &S) {
    SpatialMatrix<Type> adT;

    // T = [Rah,   0]
    //     [S  R]

    // diagonal blocks
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            adT.data(i, j) = R.data(i, j);
            adT.data(i + 3, j + 3) = R.data(i, j);
        }
    }

    // lower off diagonal
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            adT.data(i + 3, j) = S.data(i, j);
        }
    }

    return adT;
}

CUDA_CALLABLE inline int row_index(int stride, int i, int j) {
    return i * stride + j;
}

// builds spatial Jacobian J which is an (joint_count*6)x(dof_count) matrix
template<typename Type>
CUDA_CALLABLE inline void spatial_jacobian(
    const SpatialVector<Type> *S,
    const int *joint_parents,
    const int *joint_qd_start,
    int joint_start,// offset of the first joint for the articulation
    int joint_count,
    int J_start,
    Type *J) {
    const int articulation_dof_start = joint_qd_start[joint_start];
    const int articulation_dof_end = joint_qd_start[joint_start + joint_count];
    const int articulation_dof_count = articulation_dof_end - articulation_dof_start;

    // shift output pointers
    const int S_start = articulation_dof_start;

    S += S_start;
    J += J_start;

    for (int i = 0; i < joint_count; ++i) {
        const int row_start = i * 6;

        int j = joint_start + i;
        while (j != -1) {
            const int joint_dof_start = joint_qd_start[j];
            const int joint_dof_end = joint_qd_start[j + 1];
            const int joint_dof_count = joint_dof_end - joint_dof_start;

            // fill out each row of the Jacobian walking up the tree
            //for (int col=dof_start; col < dof_end; ++col)
            for (int dof = 0; dof < joint_dof_count; ++dof) {
                const int col = (joint_dof_start - articulation_dof_start) + dof;

                J[row_index(articulation_dof_count, row_start + 0, col)] = S[col].w[0];
                J[row_index(articulation_dof_count, row_start + 1, col)] = S[col].w[1];
                J[row_index(articulation_dof_count, row_start + 2, col)] = S[col].w[2];
                J[row_index(articulation_dof_count, row_start + 3, col)] = S[col].v[0];
                J[row_index(articulation_dof_count, row_start + 4, col)] = S[col].v[1];
                J[row_index(articulation_dof_count, row_start + 5, col)] = S[col].v[2];
            }

            j = joint_parents[j];
        }
    }
}

template<typename Type>
CUDA_CALLABLE inline void spatial_mass(const SpatialMatrix<Type> *I_s, int joint_start, int joint_count, int M_start, Type *M) {
    const int stride = joint_count * 6;

    for (int l = 0; l < joint_count; ++l) {
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                M[M_start + row_index(stride, l * 6 + i, l * 6 + j)] = I_s[joint_start + l](i, j);
            }
        }
    }
}

}// namespace vox