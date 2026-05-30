#pragma once

#include <vector>

#include "../core/devices.hpp"

namespace sx::Metal {
[[nodiscard]] std::vector<DevicePtr> devices();
}
