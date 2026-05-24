#pragma once

#include <cstddef>

#include "../core/device.hpp"

namespace Cpu {

class Device final : public ::Device {
public:
    [[nodiscard]] Platform platform() const noexcept override;

    [[nodiscard]] std::byte *allocate(std::size_t nbytes) const;
    void deallocate(std::byte *ptr) const noexcept;

protected:
    [[nodiscard]] std::string repr_() const override;
};

} // namespace Cpu
