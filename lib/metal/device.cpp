#include <new>
#include <stdexcept>
#include <string>

#include "device.hpp"

namespace sx::Metal {

Device::Device(std::int32_t id, MTL::Device *device) : sx::Device{id}, device_{device} {
    if (device_ == nullptr) {
        throw std::invalid_argument{"invalid device"};
    }
    device_->retain();
}

Device::~Device() noexcept {
    device_->release();
}

MTL::Buffer *Device::allocate(std::size_t nbytes) const {
    if (nbytes == 0) {
        return nullptr;
    }

    MTL::Buffer *buffer{device_->newBuffer(nbytes, MTL::ResourceStorageModeShared)};
    if (buffer == nullptr) {
        throw std::bad_alloc{};
    }

    return buffer;
}

void Device::deallocate(MTL::Buffer *buffer) const noexcept {
    if (buffer != nullptr) {
        buffer->release();
    }
}

} // namespace sx::Metal
