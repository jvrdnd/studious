#pragma once

#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>

#include "../core/device.hpp"

namespace Metal {

class Device final : public ::Device {
public:
    explicit Device(std::uint64_t id, MTL::Device *device);
    ~Device() noexcept override;

    [[nodiscard]] Platform platform() const noexcept override;
    [[nodiscard]] std::uint64_t id() const noexcept;

    [[nodiscard]] MTL::Buffer *allocate(std::size_t nbytes) const;
    void deallocate(MTL::Buffer *buffer) const noexcept;

protected:
    [[nodiscard]] std::string repr_() const override;

private:
    std::uint64_t id_;
    MTL::Device *device_;
};

} // namespace Metal
