#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <memory>

#include "../core/buffer.hpp"
#include "device.hpp"

namespace sx::Metal {

class Buffer final : public sx::Buffer<Device> {
public:
    explicit Buffer(std::shared_ptr<const Device> device, std::size_t size) :
        sx::Buffer<Device>{device}, handle_{device->allocate(size)} {}
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

[[nodiscard]] inline std::shared_ptr<const sx::Buffer<Device>>
make_buffer(std::shared_ptr<const Device> device, std::size_t size) {
    return std::make_shared<Buffer>(std::move(device), size);
}

} // namespace sx::Metal
