//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#ifndef CUDA_GTEST_PLUGIN_H_
#define CUDA_GTEST_PLUGIN_H_

#include <gtest/gtest.h>
#ifdef __CUDACC__

#ifndef CUDA_LAST_ERROR
#define CUDA_LAST_ERROR(msg)                                                    \
    {                                                                           \
        cudaDeviceSynchronize();                                                \
        cudaError_t error = cudaGetLastError();                                 \
        if (error != cudaSuccess) {                                             \
            fprintf(stderr, "ERROR: %s: %s\n", msg, cudaGetErrorString(error)); \
            exit(-1);                                                           \
        }                                                                       \
    }
#endif

struct TestTransporter {
    bool result[64];
    int evaluatedCount;

    __host__ __device__ TestTransporter() : evaluatedCount(0){};
};

__host__ __device__ inline void setTestTransporterValue(TestTransporter *transporter, bool result) {
    transporter->result[transporter->evaluatedCount] = result;
    transporter->evaluatedCount++;
}

#define CUDA_TEST_CLASS_NAME_(test_case_name, test_name) \
    kernel_##test_case_name##_##test_name##_Test

#ifdef __CUDA_ARCH__
#undef TEST
#define CUDA_DEAD_FUNCTION_NAME_(test_case_name, test_name) \
    MAKE_UNIQUE(dead_function_##test_case_name##_##test_name##_Test)
#define TEST(test_case_name, test_name) void CUDA_DEAD_FUNCTION_NAME_(test_case_name, test_name)(TestTransporter * testTransporter)//GTEST_TEST(test_case_name, test_name)
#define TESTTRANSPORTERDEFINITIONWITHCOMMA , TestTransporter *testTransporter
#define TESTTRANSPORTERDEFANDINSTANCE
#define TESTTRANSPORTERDEFINITION TestTransporter *testTransporter
#define TESTCALLHOST
#define TESTCALLDEVICE test(testTransporter)
#else
#define CUDA_DEAD_FUNCTION_NAME_(test_case_name, test_name)
#define TESTTRANSPORTERDEFANDINSTANCE TestTransporter *testTransporter = new TestTransporter;
#define TESTTRANSPORTERDEFINITIONWITHCOMMA
#define TESTTRANSPORTERDEFINITION
#define TESTCALLHOST test()
#define TESTCALLDEVICE
#endif

#define TESTKERNELCALL(test_case_name, test_name)       \
    CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) \
    test;                                               \
    CUDA_TEST_CLASS_NAME_(test_case_name, test_name)<<<1, 1>>>(test, dTestTransporter)

#define CUDA_ASSERT_EQ(expected, actual)

#ifdef __CUDA_ARCH__
#undef EXPECT_FLOAT_EQ
#define EXPECT_FLOAT_EQ(val1, val2) setTestTransporterValue(testTransporter, (val1) == (val2));

#undef EXPECT_EQ
#define EXPECT_EQ(val1, val2) setTestTransporterValue(testTransporter, (val1) == (val2));

#undef EXPECT_NEAR
#define EXPECT_NEAR(val1, val2, abs_error) setTestTransporterValue(testTransporter, abs((val1) - (val2)) < (abs_error));

#undef EXPECT_TRUE
#define EXPECT_TRUE(val1) setTestTransporterValue(testTransporter, val1);

#undef EXPECT_FALSE
#define EXPECT_FALSE(val1) setTestTransporterValue(testTransporter, !(val1));
#endif

#define CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) \
    test_function_##test_case_name##_##test_name##_Test
#define TEST_NAME_CUDA(test_name) \
    test_name##_CUDA

#define CONCATENATE_DETAIL(x, y) x##y
#define CONCATENATE(x, y) CONCATENATE_DETAIL(x, y)
#define MAKE_UNIQUE(x) CONCATENATE(x, __COUNTER__)

#define CUDA_TEST(test_case_name, test_name)                                                                                                                        \
    struct CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) {                                                                                                    \
        __host__ __device__ void operator()(TestTransporter *testTransporter);                                                                                      \
    };                                                                                                                                                              \
    __global__ void CUDA_TEST_CLASS_NAME_(test_case_name, test_name)(CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) test, TestTransporter * testTransporter);  \
    GTEST_TEST(test_case_name, test_name) {                                                                                                                         \
        CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name)                                                                                                         \
        test;                                                                                                                                                       \
        TestTransporter *testTransporter = new TestTransporter;                                                                                                     \
        test(testTransporter);                                                                                                                                      \
    };                                                                                                                                                              \
    TEST(test_case_name, test_name##_CUDA) {                                                                                                                        \
        TestTransporter *dTestTransporter;                                                                                                                          \
        cudaMalloc((void **)(&dTestTransporter), sizeof(TestTransporter));                                                                                          \
        CUDA_LAST_ERROR("malloc");                                                                                                                                  \
        TESTTRANSPORTERDEFANDINSTANCE                                                                                                                               \
        cudaMemcpy(dTestTransporter, testTransporter, sizeof(TestTransporter), cudaMemcpyHostToDevice);                                                             \
        CUDA_LAST_ERROR("memcopyhosttodevice");                                                                                                                     \
        CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name)                                                                                                         \
        test;                                                                                                                                                       \
        CUDA_TEST_CLASS_NAME_(test_case_name, test_name)<<<1, 1>>>(test, dTestTransporter);                                                                         \
        CUDA_LAST_ERROR("kernel call");                                                                                                                             \
        cudaMemcpy(testTransporter, dTestTransporter, sizeof(TestTransporter), cudaMemcpyDeviceToHost);                                                             \
        CUDA_LAST_ERROR("memcopydevicetohost");                                                                                                                     \
        for (int i = 0; i < testTransporter->evaluatedCount; i++)                                                                                                   \
            GTEST_EXPECT_TRUE(testTransporter->result[i]) << "assert statement(i = " << i << ") failed.\n";                                                         \
    };                                                                                                                                                              \
    __global__ void CUDA_TEST_CLASS_NAME_(test_case_name, test_name)(CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name) test, TestTransporter * testTransporter) { \
        test(testTransporter);                                                                                                                                      \
    }                                                                                                                                                               \
    __host__ __device__ void CUDA_TEST_FUNCTION_NAME_(test_case_name, test_name)::operator()(TestTransporter *testTransporter)
#else
#warning "To enable CUDA tests compile with nvcc"
#define CUDA_TEST(test_case_name, test_name) TEST(test_case_name, test_name)
#endif

#endif /* CUDA_GTEST_PLUGIN_H_ */