//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include <gtest/gtest.h>
#include "runtime/device.h"
#include "runtime/transform.h"

using namespace vox;

class Device1Test : public ::testing::Test {
public:
    void SetUp() override {
        vox::init();
    }

    static void launch() {
        // Print the local time from GPU threads.
        time_t cur_time;
        time(&cur_time);
        tm t = *localtime(&cur_time);

        // Define a CUDA kernel with closure. Tag it with MGPU_DEVICE and compile
        // with --expt-extended-lambda in CUDA 7.5 to run it on the GPU.
        auto k = [=] CUDA_CALLABLE_DEVICE(int index) {
            // This gets run on the GPU. Simply by referencing t.tm_year inside
            // the lambda, the time is copied from its enclosing scope on the host
            // into GPU constant memory and made available to the kernel.

            // Adjust for daylight savings.
            int hour = (t.tm_hour + (t.tm_isdst ? 0 : 11)) % 12;
            if(!hour) hour = 12;

            // Use CUDA's printf. It won't be shown until the context.synchronize()
            // is called.
            printf("Thread %d says the year is %d. The time is %d:%2d.\n",
                   index, 1900 + t.tm_year, hour, t.tm_min);
        };
        // Run kernel k with 10 GPU threads. We could even define the lambda
        // inside the first argument of transform and not even name it.
        transform(k, 10, device().stream);
    }
};

TEST_F(Device1Test, constructor) {
    std::cout << "total cuda device: " << device_count() << std::endl;
    std::cout << device(0).name() << std::endl;
    std::cout << device(0).arch() << std::endl;
    std::cout << device(0).info().ptx_version << std::endl;
}

TEST_F(Device1Test, launch) {
    Device1Test::launch();
}