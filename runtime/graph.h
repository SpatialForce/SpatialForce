//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {
class Device;

class Graph {
public:
    Graph(const Device &device, cudaGraphExec_t graph);

    ~Graph();

    void launch();

private:
    const Device &device_;
    cudaGraphExec_t graph_{};
};

/// Begin capture of a CUDA graph
//
//    Captures all subsequent kernel launches and memory operations on CUDA devices.
//    This can be used to record large numbers of kernels and replay them with low-overhead.
/// \param index The device to capture on
void capture_begin(uint32_t index = 0);

/// Ends the capture of a CUDA graph
/// \param index The device to capture on
/// \return A handle to a CUDA graph object that can be launched
Graph end_capture(uint32_t index = 0);

}// namespace vox