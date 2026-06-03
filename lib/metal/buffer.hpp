#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <memory>

#include "../core/buffer.hpp"
#include "device.hpp"

namespace sx::Metal {

class Buffer final : public sx::Buffer<Device> {
public:
    explicit Buffer(std::shared_ptr<const Device> device, std::size_t size, StorageMode mode = StorageMode::Shared) :
        sx::Buffer<Device>{device}, handle_{device->allocate(size, mode)} {}
    ~Buffer() noexcept override {
        device()->deallocate(handle_);
    }

    [[nodiscard]] std::byte *data() const noexcept override {
        return static_cast<std::byte *>(handle_->contents());
    }
    [[nodiscard]] std::size_t size() const noexcept override {
        return handle_->length();
    }

private:
    MTL::Buffer *const handle_;
};

} // namespace sx::Metal
