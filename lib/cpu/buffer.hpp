#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <nanobind/nanobind.h>

#include "../core/buffer.hpp"
#include "device.hpp"

namespace Cpu {

class Buffer final : public ::Buffer {
public:
    const std::shared_ptr<const Device> device;

    Buffer(std::shared_ptr<const Device> device, std::size_t size);
    ~Buffer() noexcept override;

    std::byte *data() const noexcept override;
    std::size_t size() const noexcept override;

private:
    std::byte *const data_;
    const std::size_t size_;
};

} // namespace Cpu
