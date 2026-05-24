#pragma once

#include <memory>
#include <vector>

#include "device.hpp"

using DevicePtr = std::shared_ptr<const Device>;
[[nodiscard]] std::vector<DevicePtr> devices();
[[nodiscard]] std::vector<DevicePtr> devices(Platform platform);
