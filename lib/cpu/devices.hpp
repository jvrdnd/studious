#pragma once

#include <vector>

#include "../core/devices.hpp"

namespace sx::Cpu {
[[nodiscard]] std::vector<DevicePtr> devices();
}
