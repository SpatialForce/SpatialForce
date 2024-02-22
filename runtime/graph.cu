//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "graph.h"
#include "device.h"

namespace vox {
Graph::Graph(const Device &device, cudaGraphExec_t graph) : device_{device}, graph_{graph} {}

Graph::~Graph() {
    ContextGuard guard(device_.primary_context());
    check_cuda(cudaGraphExecDestroy(graph_));
}

void Graph::launch() {
    ContextGuard guard(device_.primary_context());
    check_cuda(cudaGraphLaunch(graph_, device_.stream.handle()));
}

void capture_begin(uint32_t index) {
    const auto &d = device(index);
    ContextGuard guard(d.primary_context());
    check_cuda(cudaStreamBeginCapture(d.stream.handle(), cudaStreamCaptureModeGlobal));
}

Graph capture_end(uint32_t index) {
    const auto &d = device(index);
    ContextGuard guard(d.primary_context());

    cudaGraph_t graph = nullptr;
    check_cuda(cudaStreamEndCapture(d.stream.handle(), &graph));

    // enable to create debug GraphVis visualization of graph
    // cudaGraphDebugDotPrint(graph, "graph.dot", cudaGraphDebugDotFlagsVerbose);

    cudaGraphExec_t graph_exec = nullptr;
    // check_cuda(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0));

    // can use after CUDA 11.4 to permit graphs to capture cudaMallocAsync() operations
    check_cuda(cudaGraphInstantiateWithFlags(&graph_exec, graph, cudaGraphInstantiateFlagAutoFreeOnLaunch));

    // free source graph
    check_cuda(cudaGraphDestroy(graph));

    return {d, graph_exec};
}

}// namespace vox