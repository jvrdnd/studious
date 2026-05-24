#include "../cpu/devices.hpp"
#include "../metal/devices.hpp"
#include "devices.hpp"

std::vector<DevicePtr> devices() {
    std::vector<DevicePtr> devices;
    std::vector<DevicePtr> cpu_devices = Cpu::devices();
    std::vector<DevicePtr> metal_devices = Metal::devices();

    devices.reserve(cpu_devices.size() + metal_devices.size());
    devices.insert(devices.end(), cpu_devices.begin(), cpu_devices.end());
    devices.insert(devices.end(), metal_devices.begin(), metal_devices.end());

    return devices;
}

std::vector<DevicePtr> devices(Platform platform) {
    switch (platform) {
        case Platform::Cpu:
            return Cpu::devices();
        case Platform::Metal:
            return Metal::devices();
    }
}
