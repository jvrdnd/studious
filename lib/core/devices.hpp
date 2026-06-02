#pragma once

#include <vector>

#include "device.hpp"

namespace sx {

[[nodiscard]] std::vector<DevicePtr> devices();
[[nodiscard]] std::vector<DevicePtr> devices(Platform platform);

} // namespace sx
