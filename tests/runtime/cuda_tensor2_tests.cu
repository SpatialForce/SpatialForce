//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/cuda_tensor.h"
#include "runtime/cuda_tensor_view.h"

#include <gtest/gtest.h>

using namespace vox;

TEST(CudaTensor2, Constructors) {
    {
        CudaTensor2<float> arr;
        EXPECT_EQ(0u, arr.width());
        EXPECT_EQ(0u, arr.height());
    }
    {
        CudaTensor2<float> arr(CudaStdArray<size_t, 2>(3, 7));
        EXPECT_EQ(3u, arr.width());
        EXPECT_EQ(7u, arr.height());
        for (size_t i = 0; i < 21; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }
    }
    {
        CudaTensor2<float> arr(CudaStdArray<size_t, 2>(1, 9), 1.5f);
        EXPECT_EQ(1u, arr.width());
        EXPECT_EQ(9u, arr.height());
        for (size_t i = 0; i < 9; ++i) {
            EXPECT_FLOAT_EQ(1.5f, arr[i]);
        }
    }
    {
        CudaTensor2<float> arr(5, 2);
        EXPECT_EQ(5u, arr.width());
        EXPECT_EQ(2u, arr.height());
        for (size_t i = 0; i < 10; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }
    }
    {
        CudaTensor2<float> arr(3, 4, 7.f);
        EXPECT_EQ(3u, arr.width());
        EXPECT_EQ(4u, arr.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ(7.f, arr[i]);
        }
    }
    {
        CudaTensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                                {5.f, 6.f, 7.f, 8.f},
                                {9.f, 10.f, 11.f, 12.f}});
        EXPECT_EQ(4u, arr.width());
        EXPECT_EQ(3u, arr.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr[i]);
        }
    }
    {
        Tensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                            {5.f, 6.f, 7.f, 8.f},
                            {9.f, 10.f, 11.f, 12.f}});
        CudaTensor2<float> arr2(arr);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }
    {
        CudaTensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                                {5.f, 6.f, 7.f, 8.f},
                                {9.f, 10.f, 11.f, 12.f}});
        CudaTensor2<float> arr2(arr);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }
    {
        CudaTensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                                {5.f, 6.f, 7.f, 8.f},
                                {9.f, 10.f, 11.f, 12.f}});
        CudaTensorView2<float> arrVew(arr.data(), arr.shape());
        EXPECT_EQ(4u, arrVew.width());
        EXPECT_EQ(3u, arrVew.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arrVew[i]);
        }
    }
}

TEST(CudaTensor2, At) {
    {
        float values[12] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f,
                            6.f, 7.f, 8.f, 9.f, 10.f, 11.f};
        CudaTensor2<float> arr(4, 3);
        for (size_t i = 0; i < 12; ++i) {
            arr[i] = values[i];
        }

        // Test row-major
        EXPECT_FLOAT_EQ(0.f, arr(0, 0));
        EXPECT_FLOAT_EQ(1.f, arr(1, 0));
        EXPECT_FLOAT_EQ(2.f, arr(2, 0));
        EXPECT_FLOAT_EQ(3.f, arr(3, 0));
        EXPECT_FLOAT_EQ(4.f, arr(0, 1));
        EXPECT_FLOAT_EQ(5.f, arr(1, 1));
        EXPECT_FLOAT_EQ(6.f, arr(2, 1));
        EXPECT_FLOAT_EQ(7.f, arr(3, 1));
        EXPECT_FLOAT_EQ(8.f, arr(0, 2));
        EXPECT_FLOAT_EQ(9.f, arr(1, 2));
        EXPECT_FLOAT_EQ(10.f, arr(2, 2));
        EXPECT_FLOAT_EQ(11.f, arr(3, 2));
    }
}

TEST(CudaTensor2, CopyFrom) {
    // From Array
    {
        Tensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                            {5.f, 6.f, 7.f, 8.f},
                            {9.f, 10.f, 11.f, 12.f}});
        CudaTensor2<float> arr2(2, 5);

        arr2.copyFrom(arr);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From TensorView
    {
        Tensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                            {5.f, 6.f, 7.f, 8.f},
                            {9.f, 10.f, 11.f, 12.f}});
        CudaTensor2<float> arr2(2, 5);

        arr2.copyFrom(arr.view());
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From CudaTensor
    {
        CudaTensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                                {5.f, 6.f, 7.f, 8.f},
                                {9.f, 10.f, 11.f, 12.f}});
        CudaTensor2<float> arr2(2, 5);

        arr2.copyFrom(arr);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From CudaTensorView
    {
        CudaTensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                                {5.f, 6.f, 7.f, 8.f},
                                {9.f, 10.f, 11.f, 12.f}});
        CudaTensor2<float> arr2(2, 5);

        arr2.copyFrom(arr.view());
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }
}

TEST(CudaTensor2, CopyTo) {
    // To Array
    {
        CudaTensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                                {5.f, 6.f, 7.f, 8.f},
                                {9.f, 10.f, 11.f, 12.f}});
        Tensor2<float> arr2(2, 5);

        arr.copyTo(arr2);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // To TensorView
    {
        CudaTensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                                {5.f, 6.f, 7.f, 8.f},
                                {9.f, 10.f, 11.f, 12.f}});
        Tensor2<float> arr2(4, 3);
        TensorView2<float> arrView2 = arr2.view();

        arr.copyTo(arrView2);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From CudaTensor
    {
        CudaTensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                                {5.f, 6.f, 7.f, 8.f},
                                {9.f, 10.f, 11.f, 12.f}});
        CudaTensor2<float> arr2(2, 5);

        arr.copyTo(arr2);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }

    // From CudaTensorView
    {
        CudaTensor2<float> arr({{1.f, 2.f, 3.f, 4.f},
                                {5.f, 6.f, 7.f, 8.f},
                                {9.f, 10.f, 11.f, 12.f}});
        CudaTensor2<float> arr2(4, 3);
        CudaTensorView2<float> arrView2 = arr2.view();

        arr.copyTo(arrView2);
        EXPECT_EQ(4u, arr2.width());
        EXPECT_EQ(3u, arr2.height());
        for (size_t i = 0; i < 12; ++i) {
            EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
        }
    }
}

TEST(CudaTensor2, Fill) {
    CudaTensor2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});

    arr.fill(42.0f);
    EXPECT_EQ(4u, arr.width());
    EXPECT_EQ(3u, arr.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(42.0f, arr[i]);
    }
}

TEST(CudaTensor2, Resize) {
    {
        CudaTensor2<float> arr;
        arr.resize(CudaStdArray<size_t, 2>(2, 9));
        EXPECT_EQ(2u, arr.width());
        EXPECT_EQ(9u, arr.height());
        for (size_t i = 0; i < 18; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }

        arr.resize(CudaStdArray<size_t, 2>(8, 13), 4.f);
        cudaDeviceSynchronize();
        EXPECT_EQ(8u, arr.width());
        EXPECT_EQ(13u, arr.height());
        for (size_t i = 0; i < 8; ++i) {
            for (size_t j = 0; j < 13; ++j) {
                if (i < 2 && j < 9) {
                    EXPECT_FLOAT_EQ(0.f, arr(i, j));
                } else {
                    EXPECT_FLOAT_EQ(4.f, arr(i, j));
                }
            }
        }
    }
    {
        CudaTensor2<float> arr;
        arr.resize(7, 6);
        EXPECT_EQ(7u, arr.width());
        EXPECT_EQ(6u, arr.height());
        for (size_t i = 0; i < 42; ++i) {
            EXPECT_FLOAT_EQ(0.f, arr[i]);
        }

        arr.resize(1, 9, 3.f);
        EXPECT_EQ(1u, arr.width());
        EXPECT_EQ(9u, arr.height());
        for (size_t i = 0; i < 1; ++i) {
            for (size_t j = 0; j < 9; ++j) {
                if (j < 6) {
                    EXPECT_FLOAT_EQ(0.f, arr(i, j));
                } else {
                    EXPECT_FLOAT_EQ(3.f, arr(i, j));
                }
            }
        }
    }
}

TEST(CudaTensor2, Clear) {
    CudaTensor2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});

    arr.clear();
    EXPECT_EQ(0u, arr.width());
    EXPECT_EQ(0u, arr.height());
}

TEST(CudaTensor2, Swap) {
    CudaTensor2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});
    CudaTensor2<float> arr2(2, 5, 42.f);

    arr.swap(arr2);

    EXPECT_EQ(2u, arr.width());
    EXPECT_EQ(5u, arr.height());
    for (size_t i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(42.0f, arr[i]);
    }

    EXPECT_EQ(4u, arr2.width());
    EXPECT_EQ(3u, arr2.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
    }
}

TEST(CudaTensor2, View) {
    CudaTensor2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});

    auto view = arr.view();

    EXPECT_EQ(4u, view.width());
    EXPECT_EQ(3u, view.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, view[i]);
    }
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(arr(i, j), view(i, j));
        }
    }

    const auto &arrRef = arr;
    auto constView = arrRef.view();

    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, constView[i]);
    }
    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(arr(i, j), constView(i, j));
        }
    }

    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            view(i, j) = float(i + 4 * j);
        }
    }

    for (size_t j = 0; j < 3; ++j) {
        for (size_t i = 0; i < 4; ++i) {
            EXPECT_FLOAT_EQ(float(i + 4 * j), arr(i, j));
            EXPECT_FLOAT_EQ(float(i + 4 * j), constView(i, j));
        }
    }
}

TEST(CudaTensor2, AssignmentOperator) {
    CudaTensor2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});
    CudaTensor2<float> arr2(2, 5, 42.f);

    arr2 = arr;

    EXPECT_EQ(4u, arr.width());
    EXPECT_EQ(3u, arr.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, arr[i]);
    }

    EXPECT_EQ(4u, arr2.width());
    EXPECT_EQ(3u, arr2.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
    }
}

TEST(CudaTensor2, MoveOperator) {
    CudaTensor2<float> arr(
        {{1.f, 2.f, 3.f, 4.f}, {5.f, 6.f, 7.f, 8.f}, {9.f, 10.f, 11.f, 12.f}});
    CudaTensor2<float> arr2(2, 5, 42.f);

    arr2 = std::move(arr);

    EXPECT_EQ(0u, arr.width());
    EXPECT_EQ(0u, arr.height());
    EXPECT_EQ(nullptr, arr.data());
    EXPECT_EQ(4u, arr2.width());
    EXPECT_EQ(3u, arr2.height());
    for (size_t i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ((float)i + 1.f, arr2[i]);
    }
}
