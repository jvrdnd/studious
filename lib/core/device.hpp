#pragma once

#include <cstddef>
#include <string>

#include "platform.hpp"

namespace sx {

class Device {
public:
    virtual ~Device() noexcept = default;

    Device(const Device &) = delete;
    Device &operator=(const Device &) = delete;
    Device(Device &&) = delete;
    Device &operator=(Device &&) = delete;

    [[nodiscard]] virtual Platform platform() const noexcept = 0;

    [[nodiscard]] std::string repr() const;

protected:
    explicit Device() = default;

    [[nodiscard]] virtual std::string repr_() const = 0;
};

using DevicePtr = std::shared_ptr<const Device>;

} // namespace sx
