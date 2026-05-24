#pragma once

#include <cstddef>
#include <memory>

#include "./device.hpp"

class Buffer {
public:
    const std::shared_ptr<const Device> device;

    virtual ~Buffer() noexcept = default;

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(Buffer &&) = delete;

    virtual std::byte *data() const noexcept = 0;
    virtual std::size_t size() const noexcept = 0;

protected:
    Buffer(std::shared_ptr<const Device> device);
};
