//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "cuda_util.h"
#include "device.h"
#include "event.h"
#include "stream.h"

namespace vox {
Stream::Stream(Device &device, std::optional<CUstream> cuda_stream) : device_{device}, context_{device.primary_context()} {
    if (!cuda_stream) {
        ContextGuard guard(context_);
        check_cu(cuStreamCreate(&handle_, CU_STREAM_DEFAULT));
        owner_ = true;
    } else {
        handle_ = cuda_stream.value();
        owner_ = false;
    }
}

Stream::Stream(Stream &&stream) noexcept
    : owner_{stream.owner_},
      handle_{stream.handle_},
      device_{stream.device_},
      context_{stream.context_} {
    stream.owner_ = false;
    stream.handle_ = nullptr;
}

Stream::~Stream() {
    if (owner_) {
        ContextGuard guard(context_);
        check_cu(cuStreamDestroy(static_cast<CUstream>(handle_)));
    }
}

void Stream::record_event(Event &event) const {
    check_cu(cuEventRecord(event.handle(), handle_));
}

void Stream::wait_event(Event &event) const {
    check_cu(cuStreamWaitEvent(handle_, event.handle(), 0));
}

void Stream::wait_stream(Stream &other_stream, Event &event) const {
    check_cu(cuEventRecord(event.handle(), other_stream.handle_));
    check_cu(cuStreamWaitEvent(handle_, event.handle(), 0));
}

void Stream::synchronize() const {
    ContextGuard guard(context_);
    check_cu(cuStreamSynchronize(handle()));
}

CUstream Stream::handle() const { return handle_; }

void Stream::memcpy_h2d(void *dest, void *src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyHostToDevice, handle_));
}
void Stream::memcpy_d2h(void *dest, void *src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToHost, handle_));
}
void Stream::memcpy_d2d(void *dest, void *src, size_t n) {
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDeviceToDevice, handle_));
}

void Stream::memcpy_peer(void *dest, void *src, size_t n) {
    // NB: assumes devices involved support UVA
    check_cuda(cudaMemcpyAsync(dest, src, n, cudaMemcpyDefault, handle_));
}

void Stream::memset(void *dest, int value, size_t n) {
    check_cuda(cudaMemsetAsync(dest, value, n, handle_));
}

const Stream &stream(uint32_t index) {
    return device(index).stream;
}

}// namespace vox
