//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "runtime/cuda_tensor.h"
#include "runtime/cuda_tensor_view.h"

#include <gtest/gtest.h>

using namespace vox;

class CudaTensor1Test : public ::testing::Test {
public:
    void SetUp() override {
        vox::init();
    }
};

TEST_F(CudaTensor1Test, Constructors) {
    CudaTensor1<float> arr0;
    EXPECT_EQ(0u, arr0.length());

    CudaTensor1<float> arr1(9, 1.5f);
    EXPECT_EQ(9u, arr1.length());
    for (size_t i = 0; i < arr1.length(); ++i) {
        EXPECT_FLOAT_EQ(1.5f, arr1[i]);
    }

    Tensor1<float> arr9({1.0f, 2.0f, 3.0f});
    TensorView1<float> view9(arr9);
    CudaTensor1<float> arr10(view9);
    EXPECT_EQ(3u, arr10.length());
    for (size_t i = 0; i < arr10.length(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr10[i]);
    }
    CudaTensor1<float> arr11(arr9.view());
    EXPECT_EQ(3u, arr11.length());
    for (size_t i = 0; i < arr11.length(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr11[i]);
    }

    CudaTensor1<float> arr2(arr1.view());
    EXPECT_EQ(9u, arr2.length());
    for (size_t i = 0; i < arr2.length(); ++i) {
        EXPECT_FLOAT_EQ(1.5f, arr2[i]);
    }

    CudaTensor1<float> arr3({1.0f, 2.0f, 3.0f});
    EXPECT_EQ(3u, arr3.length());
    for (size_t i = 0; i < arr3.length(); ++i) {
        float a = arr3[i];
        EXPECT_FLOAT_EQ(1.0f + i, arr3[i]);
    }

    CudaTensor1<float> arr8(std::vector<float>{1.0f, 2.0f, 3.0f});
    EXPECT_EQ(3u, arr8.length());
    for (size_t i = 0; i < arr8.length(); ++i) {
        EXPECT_FLOAT_EQ(1.0f + i, arr8[i]);
    }

    CudaTensor1<float> arr6(arr8);
    EXPECT_EQ(3u, arr6.length());
    for (size_t i = 0; i < arr8.length(); ++i) {
        EXPECT_FLOAT_EQ(arr8[i], arr6[i]);
    }

    CudaTensor1<float> arr7 = std::move(arr6);
    EXPECT_EQ(3u, arr7.length());
    EXPECT_EQ(0u, arr6.length());
    for (size_t i = 0; i < arr6.length(); ++i) {
        EXPECT_FLOAT_EQ(arr6[i], arr7[i]);
    }
}

TEST_F(CudaTensor1Test, CopyFrom) {
    // Copy from std::vector
    CudaTensor1<float> arr1;
    std::vector<float> vec({1, 2, 3, 4, 5, 6, 7, 8, 9});
    arr1.copyFrom(vec);

    EXPECT_EQ(vec.size(), arr1.length());
    for (size_t i = 0; i < arr1.length(); ++i) {
        EXPECT_FLOAT_EQ(vec[i], arr1[i]);
    }

    // Copy from CPU Array
    CudaTensor1<float> arr2;
    Tensor1<float> cpuArr({1, 2, 3, 4, 5, 6, 7, 8, 9});
    arr2.copyFrom(cpuArr);

    EXPECT_EQ(cpuArr.length(), arr2.length());
    for (size_t i = 0; i < arr2.length(); ++i) {
        EXPECT_FLOAT_EQ(cpuArr[i], arr2[i]);
    }

    // Copy from CPU TensorView
    CudaTensor1<float> arr3;
    TensorView1<float> cpuArrView = cpuArr.view();
    arr3.copyFrom(cpuArrView);

    EXPECT_EQ(cpuArrView.length(), arr3.length());
    for (size_t i = 0; i < arr3.length(); ++i) {
        EXPECT_FLOAT_EQ(cpuArrView[i], arr3[i]);
    }

    // Copy from CPU ConstTensorView
    CudaTensor1<float> arr4;
    ConstTensorView1<float> constCpuArrView = cpuArr.view();
    arr4.copyFrom(constCpuArrView);

    EXPECT_EQ(constCpuArrView.length(), arr4.length());
    for (size_t i = 0; i < arr4.length(); ++i) {
        EXPECT_FLOAT_EQ(constCpuArrView[i], arr4[i]);
    }

    // Copy from CudaTensor
    CudaTensor1<float> arr5;
    CudaTensor1<float> cudaArr({1, 2, 3, 4, 5, 6, 7, 8, 9});
    arr5.copyFrom(cudaArr);

    EXPECT_EQ(cudaArr.length(), arr5.length());
    for (size_t i = 0; i < arr5.length(); ++i) {
        EXPECT_FLOAT_EQ(cudaArr[i], arr5[i]);
    }

    // Copy from CudaTensorView
    CudaTensor1<float> arr6;
    CudaTensorView1<float> cudaArrView = arr6.view();
    arr6.copyFrom(cudaArrView);

    EXPECT_EQ(cudaArrView.length(), arr6.length());
    for (size_t i = 0; i < arr6.length(); ++i) {
        EXPECT_FLOAT_EQ(cudaArrView[i], arr6[i]);
    }

    // Copy from ConstCudaTensorView
    CudaTensor1<float> arr7;
    ConstCudaTensorView1<float> constCudaArrView = arr7.view();
    arr7.copyFrom(constCudaArrView);

    EXPECT_EQ(constCudaArrView.length(), arr7.length());
    for (size_t i = 0; i < arr7.length(); ++i) {
        EXPECT_FLOAT_EQ(constCudaArrView[i], arr7[i]);
    }
}

//TEST(CudaTensor1, Append) {
//    // Cuda + scalar
//    {
//        CudaTensor1<float> arr1({1.0f, 2.0f, 3.0f});
//        arr1.append(4.0f);
//        arr1.append(5.0f);
//        EXPECT_EQ(5u, arr1.length());
//        for (size_t i = 0; i < arr1.length(); ++i) {
//            float a = arr1[i];
//            EXPECT_FLOAT_EQ(1.0f + i, arr1[i]);
//        }
//    }
//
//    // Cuda + Cuda
//    {
//        CudaTensor1<float> arr1({1.0f, 2.0f, 3.0f});
//        CudaTensor1<float> arr2({4.0f, 5.0f});
//        arr1.append(arr2);
//        EXPECT_EQ(5u, arr1.length());
//        for (size_t i = 0; i < arr1.length(); ++i) {
//            float a = arr1[i];
//            EXPECT_FLOAT_EQ(1.0f + i, arr1[i]);
//        }
//    }
//
//    // Cuda + Cpu
//    {
//        CudaTensor1<float> arr1({1.0f, 2.0f, 3.0f});
//        Tensor1<float> arr2({4.0f, 5.0f});
//        arr1.append(arr2);
//        EXPECT_EQ(5u, arr1.length());
//        for (size_t i = 0; i < arr1.length(); ++i) {
//            float a = arr1[i];
//            EXPECT_FLOAT_EQ(1.0f + i, arr1[i]);
//        }
//    }
//}

TEST_F(CudaTensor1Test, View) {
    CudaTensor1<float> arr(15, 3.14f);
    CudaTensorView1<float> view = arr.view();
    EXPECT_EQ(15u, view.length());
    EXPECT_EQ(arr.data(), view.data());
    for (size_t i = 0; i < 15; ++i) {
        float val = arr[i];
        EXPECT_FLOAT_EQ(3.14f, val);
    }
}
