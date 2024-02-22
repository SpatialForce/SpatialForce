//  Copyright (c) 2024 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#include "event.h"

namespace vox {
Event::Event(Device &device, bool enable_timing) : device_{device} {
    ContextGuard guard(device_.primary_context());

    int flags = CU_EVENT_DEFAULT;
    if (!enable_timing) {
        flags |= CU_EVENT_DISABLE_TIMING;
    }

    check_cu(cuEventCreate(&event_, flags));
}

Event::~Event() {
    ContextGuard guard(device_.primary_context());
    check_cu(cuEventDestroy(event_));
}

void record_event(Event &event) {
    stream().record_event(event);
}

void wait_event(Event &event) {
    stream().wait_event(event);
}

void wait_stream(Stream &other_stream, Event &event) {
    stream().wait_stream(other_stream, event);
}

}// namespace vox