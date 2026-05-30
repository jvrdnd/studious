#pragma once

#include <concepts>
#include <cstddef>
#include <memory>
#include <nanobind/nanobind.h>
#include <utility>

#include "device.hpp"
#include "dtype.hpp"

namespace sx {

template <typename D>
    requires std::derived_from<D, Device>
class Buffer {
public:
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

std::size_t memcpy(std::byte *ptr, nanobind::handle data, DType dtype);

} // namespace sx
