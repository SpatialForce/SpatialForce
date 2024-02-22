//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

#include "device.h"

namespace vox {
class Event {
public:
    explicit Event(Device &device, bool enable_timing = false);

    ~Event();

    CUevent handle() { return event_; }

private:
    Device &device_;
    CUevent event_{};
};

/// Record a CUDA event on the current stream.
/// \param Event Event to record.
void record_event(Event &Event);

/// Make the current stream wait for a CUDA event.
/// \param event Event to wait for.
void wait_event(Event &event);

/// Make the current stream wait for another CUDA stream to complete its work.
/// \param other_stream
/// \param event Event to be used.
void wait_stream(Stream &other_stream, Event &event);

}// namespace vox