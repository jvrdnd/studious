#include <cstddef>
#include <functional>
#include <memory>
#include <nanobind/nanobind.h>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "../cpu/array.hpp"
#include "../cpu/buffer.hpp"
#include "array.hpp"
#include "dtype.hpp"
#include "platform.hpp"

namespace nb = nanobind;

namespace {

std::vector<std::size_t> infer_shape(nb::handle data) {
    std::vector<std::size_t> shape;
    while (nb::isinstance<nb::list>(data)) {
        auto list = nb::borrow<nb::list>(data);
        shape.push_back((list.size()));
        if (list.size() == 0) {
            break;
        }
        data = list[0];
    }
    return shape;
}

std::optional<DType> infer_dtype(nb::handle data, const std::vector<std::size_t> &shape, std::size_t depth = 0) {
    if (depth == shape.size()) {
        if (nb::isinstance<nb::bool_>(data)) {
            return DType::Bool;
        }
        if (nb::isinstance<nb::int_>(data)) {
            return DType::Int32;
        }
        if (nb::isinstance<nb::float_>(data)) {
            return DType::Float32;
        }
        throw nb::type_error("invalid array scalar");
    }

    if (!nb::isinstance<nb::list>(data)) {
        throw nb::type_error("data must be array or scalar");
    }

    auto list{nb::borrow<nb::list>(data)};
    if (shape[depth] != list.size()) {
        throw nb::type_error("invalid array shape");
    }

    std::optional<DType> dtype;
    for (std::size_t i{0}; i < list.size(); i++) {
        std::optional<DType> item_dtype{infer_dtype(list[i], shape, depth + 1)};
        if (dtype == DType::Float32 || item_dtype == DType::Float32) {
            dtype = DType::Float32;
        } else if (dtype == DType::Int32 || item_dtype == DType::Int32) {
            dtype = DType::Int32;
        } else if (dtype == DType::Bool || item_dtype == DType::Bool) {
            dtype = DType::Bool;
        }
    }

    return dtype;
}

} // namespace

std::shared_ptr<Array> make_array(nb::handle data, std::optional<DType> dtype, std::shared_ptr<const Device> device) {
    std::vector<std::size_t> shape{infer_shape(data)};
    std::size_t length = std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<std::size_t>{});
    DType dtype_value{infer_dtype(data, shape).value_or(DType::Float32)};
    dtype_value = dtype.value_or(dtype_value);

    switch (device->platform()) {
        case Platform::Cpu: {
            std::shared_ptr<const Cpu::Buffer> buffer = std::make_shared<Cpu::Buffer>(
                std::dynamic_pointer_cast<const Cpu::Device>(device),
                dtype_size(dtype_value) * length
            );
            memcpy(buffer->data(), data, Cpu::dispatch_dtype, dtype_value);
            return std::make_shared<Cpu::Array>(buffer, dtype_value, std::move(shape));
        }
        case Platform::Metal:
            break; // ...
    }

    throw std::invalid_argument{"invalid backend"};
}
