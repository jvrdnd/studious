#pragma once

#include <cstddef>

#include "../core/array.hpp"
#include "../core/dtype.hpp"
#include "buffer.hpp"
#include "dtype.hpp"

namespace Cpu {

template <NativeType T> class Array final : public ::Array<T> {
public:
    Array(std::shared_ptr<const Buffer> buffer) noexcept : buffer_{ std::move(buffer) }, dtype_{ to_dtype<T>() } {};

    template <typename R>
        requires Range<R> && std::same_as<std::ranges::range_value_t<R>, T>
    Array(std::shared_ptr<const Device> device, const R &range) :
        buffer_{ make_buffer(std::move(device), range) }, dtype_{ to_dtype<T>() } {}

    [[nodiscard]] std::span<const T> span() const override {
        return { reinterpret_cast<const T *>(buffer_->data()), buffer_->size() / sizeof(T) };
    }

private:
    const std::shared_ptr<const Buffer> buffer_;
    const DType dtype_;
};

} // namespace Cpu
