//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {
class Device;
class Event;

class Stream {
public:
    explicit Stream(Device &device, CUstream cuda_stream = nullptr);

    ~Stream();

    void record_event(Event &event);

    void wait_event(Event &event);

    void wait_stream(Stream &other_stream, Event &event);

    CUstream handle();

    Device &device() { return device_; }

public:
    void memcpy_h2d(void *dest, void *src, size_t n);
    void memcpy_d2h(void *dest, void *src, size_t n);
    void memcpy_d2d(void *dest, void *src, size_t n);
    void memcpy_peer(void *dest, void *src, size_t n);
    void memset(void *dest, int value, size_t n);

private:
    bool owner_{};
    Device &device_;
    CUstream handle_{};
};
}// namespace vox