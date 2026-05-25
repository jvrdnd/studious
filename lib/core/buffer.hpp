#pragma once

#include <cstddef>

class Buffer {
public:
    virtual ~Buffer() noexcept = default;

    Buffer(const Buffer &) = delete;
    Buffer &operator=(const Buffer &) = delete;
    Buffer(Buffer &&) = delete;
    Buffer &operator=(Buffer &&) = delete;

    virtual std::byte *data() const noexcept = 0;
    virtual std::size_t size() const noexcept = 0;

protected:
    Buffer() = default;
};
