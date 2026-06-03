#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>

#include "../core/device.hpp"

namespace sx::Metal {

class Device final : public sx::Device {
public:
    explicit Device(std::int32_t id, MTL::Device *device);
    ~Device() noexcept override;

    [[nodiscard]] Platform platform() const noexcept override {
        return Platform::Metal;
    }

    [[nodiscard]] MTL::Buffer *allocate(std::size_t nbytes) const;
    void deallocate(MTL::Buffer *buffer) const noexcept;

private:
    MTL::Device *const device_;
};

} // namespace sx::Metal
