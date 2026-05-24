#pragma once

#include <vector>

#include "../core/devices.hpp"

namespace Metal {
[[nodiscard]] std::vector<DevicePtr> devices();
}
