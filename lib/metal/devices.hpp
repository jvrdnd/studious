#pragma once

#include <vector>

#include "../core/device.hpp"

namespace sx::Metal {
[[nodiscard]] std::vector<DevicePtr> devices();
}
