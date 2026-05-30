#pragma once

#include <cstddef>
#include <string>

#include "../core/device.hpp"

namespace sx::Cpu {

class Device final : public sx::Device {
public:
    explicit Device() = default;

    [[nodiscard]] Platform platform() const noexcept override {
        return Platform::Cpu;
    }

    [[nodiscard]] std::byte *allocate(std::size_t size) const;
    void deallocate(std::byte *ptr) const noexcept;

protected:
    [[nodiscard]] std::string repr_() const override;
};

} // namespace sx::Cpu
