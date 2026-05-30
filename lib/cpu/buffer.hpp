#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <nanobind/nanobind.h>

#include "../core/buffer.hpp"
#include "device.hpp"

namespace sx::Cpu {

class Buffer final : public sx::Buffer<Device> {
public:
    explicit Buffer(std::shared_ptr<const Device> device, std::size_t size) :
        sx::Buffer<Device>{device}, data_{device->allocate(size)}, size_{size} {}
    ~Buffer() noexcept override {
        device()->deallocate(data_);
    }

    [[nodiscard]] std::byte *data() const noexcept override {
        return data_;
    }
    [[nodiscard]] std::size_t size() const noexcept override {
        return size_;
    }

private:
    std::byte *const data_;
    const std::size_t size_;
};

[[nodiscard]] inline std::shared_ptr<const sx::Buffer<Device>>
make_buffer(std::shared_ptr<const Device> device, std::size_t size) {
    return std::make_shared<Buffer>(std::move(device), size);
}

} // namespace sx::Cpu
