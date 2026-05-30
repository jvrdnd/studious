#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>

#include "../core/device.hpp"

namespace sx::Metal {

class Device final : public sx::Device {
public:
    explicit Device(std::uint64_t id, MTL::Device *device);
    ~Device() noexcept override;

    [[nodiscard]] Platform platform() const noexcept override {
        return Platform::Metal;
    }
    [[nodiscard]] std::uint64_t id() const noexcept {
        return id_;
    }

    [[nodiscard]] MTL::Buffer *allocate(std::size_t nbytes) const;
    void deallocate(MTL::Buffer *buffer) const noexcept;

protected:
    [[nodiscard]] std::string repr_() const override;

private:
    const std::uint64_t id_;
    MTL::Device *const device_;
};

} // namespace sx::Metal
