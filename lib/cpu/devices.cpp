#include <memory>
#include <vector>

#include "device.hpp"
#include "devices.hpp"

std::vector<DevicePtr> Cpu::devices() {
    std::vector<DevicePtr> devices;
    devices.push_back(std::make_shared<Cpu::Device>());
    return devices;
}
