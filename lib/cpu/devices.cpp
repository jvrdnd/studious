#include <memory>
#include <vector>

#include "device.hpp"
#include "devices.hpp"

namespace sx::Cpu {

std::vector<DevicePtr> devices() {
    std::vector<DevicePtr> devices;
    devices.push_back(std::make_shared<Device>());
    return devices;
}

} // namespace sx::Cpu
