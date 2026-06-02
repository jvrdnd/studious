#pragma once

#include <vector>

#include "../core/device.hpp"

namespace sx::Cpu {
[[nodiscard]] std::vector<DevicePtr> devices();
}
