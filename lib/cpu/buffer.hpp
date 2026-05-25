#pragma once

#include <cstddef>
#include <cstring>
#include <memory>
#include <ranges>

#include "../core/buffer.hpp"
#include "../core/dtype.hpp"
#include "device.hpp"
#include "dtype.hpp"

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

// support span and vector
template <typename R>
    requires Range<R> && NativeType<std::ranges::range_value_t<R>>
std::shared_ptr<const Buffer> make_buffer(std::shared_ptr<const Device> device, const R &range) {
    const std::size_t bytes{ std::ranges::size(range) * sizeof(std::ranges::range_value_t<R>) };
    const std::shared_ptr<Buffer> buffer{ std::make_shared<Buffer>(std::move(device), bytes) };
    if (bytes != 0) {
        std::memcpy(buffer->data(), std::ranges::data(range), bytes);
    }
    return buffer;
}

} // namespace Cpu
