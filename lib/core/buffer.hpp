#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <nanobind/nanobind.h>
#include <utility>

#include "device.hpp"

namespace sx {

template <typename D>
    requires std::derived_from<D, Device>
class Buffer {
public:
    using device_type = D;

    virtual ~Buffer() noexcept = default;

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(Buffer &&) = delete;

    [[nodiscard]] std::shared_ptr<const D> device() const noexcept {
        return device_;
    }

    [[nodiscard]] virtual std::byte *data() const noexcept = 0;
    [[nodiscard]] virtual std::size_t size() const noexcept = 0;

protected:
    explicit Buffer(std::shared_ptr<const D> device) noexcept : device_{std::move(device)} {}

private:
    const std::shared_ptr<const D> device_;
};

template <typename T> std::size_t memcpy(std::byte *ptr, nanobind::handle data) {
    std::size_t offset;

    if (nanobind::isinstance<nanobind::list>(data)) {
        offset = 0;
        for (nanobind::handle item : nanobind::borrow<nanobind::list>(data)) {
            offset += memcpy<T>(ptr + offset, item);
        }

    } else {
        const T scalar = nanobind::cast<T>(data);
        offset = sizeof(T);
        std::memcpy(ptr, &scalar, offset);
    }

    return offset;
}

} // namespace sx
