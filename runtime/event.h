//  Copyright (c) 2023 Feng Yang
//
//  I am making my contributions/submissions to this project solely in my
//  personal capacity and am not conveying any rights to any intellectual
//  property of any third parties.

#pragma once

namespace vox {
class Event {
public:
    explicit Event(bool enable_timing = false);

    ~Event();

    CUevent handle() { return event_; }

private:
    CUevent event_{};
};
}// namespace vox