#pragma once

#include <cstddef>
#include <nanobind/nanobind.h>
#include <utility>

#include "dtype.hpp"

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

namespace nb = nanobind;

template <typename DispatchDType>
std::size_t memcpy(std::byte *ptr, nb::handle data, DispatchDType &&dispatch_dtype, DType dtype) {
    std::size_t offset;

    if (nb::isinstance<nb::list>(data)) {
        offset = 0;
        for (nb::handle item : nb::borrow<nb::list>(data)) {
            offset += memcpy(ptr + offset, item, std::forward<DispatchDType>(dispatch_dtype), dtype);
        }

    } else {
        dispatch_dtype(dtype, [&]<typename T>() {
            T scalar = nb::cast<T>(data);
            offset = sizeof(T);
            std::memcpy(ptr, &scalar, offset);
        });
    }

    return offset;
}
