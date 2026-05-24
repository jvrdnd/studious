#include <new>
#include <stdexcept>
#include <string>

#include "device.hpp"

Metal::Device::Device(std::uint64_t id, MTL::Device *device) : id_{ id }, device_{ device } {
    if (device_ == nullptr) {
        throw std::invalid_argument{ "invalid device" };
    }
    device_->retain();
}

Metal::Device::~Device() noexcept {
    if (device_ != nullptr) {
        device_->release();
    }
}

Platform Metal::Device::platform() const noexcept {
    return Platform::Metal;
}
std::uint64_t Metal::Device::id() const noexcept {
    return id_;
}

MTL::Buffer *Metal::Device::allocate(std::size_t nbytes) const {
    if (nbytes == 0) {
        return nullptr;
    }

    MTL::Buffer *buffer{ device_->newBuffer(nbytes, MTL::ResourceStorageModeShared) };
    if (buffer == nullptr) {
        throw std::bad_alloc{};
    }

    return buffer;
}

void Metal::Device::deallocate(MTL::Buffer *buffer) const noexcept {
    if (buffer != nullptr) {
        buffer->release();
    }
}

std::string Metal::Device::repr_() const {
    return "id=" + std::to_string(id_);
}
