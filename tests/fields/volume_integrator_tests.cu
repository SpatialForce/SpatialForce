//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "fields/io/io_mesh.h"
#include "fields/io/io_mesh_1d.h"
#include "fields/io/gmsh2d_io.h"
#include "fields/volume_integrator.h"
#include "buffer.h"
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

using namespace vox::fields;
using namespace vox;

namespace {
template<typename TYPE>
struct VolumeIntegratorFunctor {
    CUDA_CALLABLE VolumeIntegratorFunctor(const mesh_t<TYPE::dim, TYPE::dim> &mesh, vox::array_t<float> output)
        : output(output) {
        integrator.mesh = mesh;
    }

    struct IntegratorFunctor {
        using RETURN_TYPE = float;
        CUDA_CALLABLE float operator()(vox::vec_t<float, TYPE::dim> pt) {
            return 1.f;
        }
    };

    inline CUDA_CALLABLE void operator()(size_t i) {
        output[i] = integrator(i, IntegratorFunctor());
    }

private:
    vox::array_t<float> output;
    VolumeIntegrator<TYPE, 2> integrator;
};
}// namespace

TEST(VolumeIntegratorTest, 1D) {
    constexpr uint32_t dim = 1;

    IOMesh1D loader(0, 1, 100);
    auto mesh = loader.create_mesh();
    mesh.sync_h2d();

    HostDeviceVector<float> result;
    result.resize(mesh.n_geometry(dim));
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(mesh.n_geometry(dim)),
                     VolumeIntegratorFunctor<Interval>(mesh.view(), result.view()));

    result.sync_d2h();
    for (int i = 0; i < mesh.n_geometry(dim); i++) {
        EXPECT_NEAR(result[i], 0.01, 1.0e-7);
    }
}

TEST(VolumeIntegratorTest, 2D) {
    constexpr uint32_t dim = 2;

    GmshMesh2D loader("grids/2d/diagsquare.msh");
    auto mesh = loader.create_mesh();
    mesh.sync_h2d();

    HostDeviceVector<float> result;
    result.resize(mesh.n_geometry(dim));
    thrust::for_each(thrust::counting_iterator<size_t>(0),
                     thrust::counting_iterator<size_t>(mesh.n_geometry(dim)),
                     VolumeIntegratorFunctor<Triangle>(mesh.view(), result.view()));

    result.sync_d2h();
    for (int i = 0; i < mesh.n_geometry(dim); i++) {
        EXPECT_NEAR(result[i], 0.005, 1.0e-7);
    }
}