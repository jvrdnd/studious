#pragma once

#include <vector>

#include "../core/devices.hpp"

namespace Cpu {
[[nodiscard]] std::vector<DevicePtr> devices();
}
