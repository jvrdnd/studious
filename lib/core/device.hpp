#pragma once

#include <cstddef>

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
    [[nodiscard]] std::int32_t id() const noexcept {
        return id_;
    }

protected:
    explicit Device(std::int32_t id) noexcept : id_{id} {};

private:
    const std::int32_t id_;
};

using DevicePtr = std::shared_ptr<const Device>;

} // namespace sx
