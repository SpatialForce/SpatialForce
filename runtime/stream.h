//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include <optional>

namespace vox {
class Device;
class Event;

class Stream {
public:
    explicit Stream(Device &device, std::optional<CUstream> cuda_stream = std::nullopt);
    Stream(Stream &&stream) noexcept;

    ~Stream();

    void record_event(Event &event) const;

    void wait_event(Event &event) const;

    void wait_stream(Stream &other_stream, Event &event) const;

    [[nodiscard]] CUstream handle() const;

    [[nodiscard]] const Device &device() const { return device_; }

    /// Manually synchronize the calling CPU thread with any outstanding CUDA work on the specified stream.
    void synchronize() const;

public:
    void memcpy_h2d(void *dest, void *src, size_t n);
    void memcpy_d2h(void *dest, void *src, size_t n);
    void memcpy_d2d(void *dest, void *src, size_t n);
    void memcpy_peer(void *dest, void *src, size_t n);
    void memset(void *dest, int value, size_t n);

private:
    bool owner_{};
    Device &device_;
    CUcontext context_{};
    CUstream handle_{};
};

/// Return the stream currently used by the given device
/// \param index device index
/// \return Stream
const Stream &stream(uint32_t index = 0);

}// namespace vox