#pragma once

#include <concepts>
#include <memory>
#include <nanobind/nanobind.h>
#include <numeric>
#include <span>
#include <vector>

#include "buffer.hpp"
#include "device.hpp"
#include "dtype.hpp"

namespace sx {

template <typename B>
    requires(requires {
        typename B::device_type;
    } && std::derived_from<typename B::device_type, Device> && std::derived_from<B, Buffer<typename B::device_type>>)
class Array {
public:
    using buffer_type = B;

    Array(DType dtype, std::vector<std::size_t> shape, std::shared_ptr<const B> buffer) :
        dtype_{dtype}, shape_{std::move(shape)}, buffer_{std::move(buffer)} {}
    virtual ~Array() noexcept = default;

    [[nodiscard]] DType dtype() const noexcept {
        return dtype_;
    }
    [[nodiscard]] std::span<const std::size_t> shape() const noexcept {
        return shape_;
    }

    [[nodiscard]] std::shared_ptr<const B> buffer() const noexcept {
        return buffer_;
    }

private:
    const DType dtype_;
    const std::vector<std::size_t> shape_;

    const std::shared_ptr<const B> buffer_;
};

[[nodiscard]] std::vector<std::size_t> infer_nb_shape(nanobind::handle data);
[[nodiscard]] std::optional<DType>
infer_nb_dtype(nanobind::handle data, const std::vector<std::size_t> &shape, std::size_t depth = 0);

template <typename D, typename B>
    requires(
        requires { typename B::device_type; } &&
        std::same_as<D, typename B::device_type> &&
        std::derived_from<D, Device> &&
        std::derived_from<B, sx::Buffer<D>>
    )
[[nodiscard]] std::shared_ptr<Array<B>>
make_array(nanobind::handle data, std::optional<DType> dtype, std::shared_ptr<const D> device) {
    const std::vector<std::size_t> shape = infer_nb_shape(data);
    const std::size_t size =
        std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{});
    const DType dtype_required = dtype.value_or(infer_nb_dtype(data, shape).value_or(DType::Float32));

    const std::shared_ptr<B> buffer = std::make_shared<B>(device, dtype_size(dtype_required) * size);
    memcpy(buffer->data(), data, dtype_required);

    return std::make_shared<Array<B>>(dtype_required, std::move(shape), buffer);
}

} // namespace sx
