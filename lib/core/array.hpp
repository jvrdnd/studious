#pragma once

#include <concepts>
#include <memory>
#include <span>
#include <vector>

#include "buffer.hpp"
#include "device.hpp"
#include "dtype.hpp"

namespace sx {

struct ArraySpec {
    DType dtype;
    std::size_t size;
    std::vector<std::size_t> shape;
};
ArraySpec infer_array_spec(nb::handle data, std::optional<DType> dtype);

template <typename D>
    requires std::derived_from<D, Device>
class Array {
public:
    Array(DType dtype, std::size_t size, std::vector<std::size_t> shape, std::shared_ptr<const Buffer<D>> buffer) :
        dtype_{dtype}, size_{size}, shape_{std::move(shape)}, buffer_{std::move(buffer)} {}
    virtual ~Array() noexcept = default;

    DType dtype() const noexcept {
        return dtype_;
    }
    std::size_t size() const noexcept {
        return size_;
    }
    std::span<const std::size_t> shape() const noexcept {
        return shape_;
    }

    [[nodiscard]] std::shared_ptr<const Buffer<D>> buffer() const noexcept {
        return buffer_;
    }

private:
    const DType dtype_;
    const std::size_t size_;
    const std::vector<std::size_t> shape_;

    const std::shared_ptr<const Buffer<D>> buffer_;
};

} // namespace sx
