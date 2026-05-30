#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include <memory>
#include <vector>

#include "device.hpp"
#include "devices.hpp"

namespace sx::Metal {

std::vector<DevicePtr> devices() {
    std::vector<DevicePtr> devices;

    NS::SharedPtr<NS::AutoreleasePool> pool{NS::TransferPtr(NS::AutoreleasePool::alloc()->init())};
    NS::SharedPtr<NS::Array> mtl_devices{NS::TransferPtr(MTL::CopyAllDevices())};

    if (mtl_devices) {
        devices.reserve(mtl_devices->count());

        for (NS::UInteger i{0}; i < mtl_devices->count(); ++i) {
            auto *mtl_device{static_cast<MTL::Device *>(mtl_devices->object(i))};
            devices.push_back(std::make_shared<Device>(mtl_device->registryID(), mtl_device));
        }
    }

    return devices;
}

} // namespace sx::Metal
