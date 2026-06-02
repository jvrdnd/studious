#pragma once

#include <cstddef>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <variant>
#include <vector>

#include "../core/array.hpp"
#include "../core/dtype.hpp"
#include "../core/platform.hpp"
#include "../metal/device.hpp"

namespace sx {

namespace Dlpack {

[[nodiscard]] DevicePtr get_device(nanobind::object data);
[[nodiscard]] Dtype get_dtype(nanobind::object data);
[[nodiscard]] std::vector<std::size_t> get_shape(nanobind::object data);
[[nodiscard]] std::vector<std::ptrdiff_t> get_strides(nanobind::object data);
[[nodiscard]] std::variant<std::vector<bool>, std::vector<std::int32_t>, std::vector<float>>
get_data(nanobind::object data);

template <BufferType B> nanobind::capsule make(nanobind::handle data) {
    const auto *array = nanobind::cast<Array<B> *>(data);
    const Dtype dtype = array->dtype();

    struct {
        std::int32_t type;
        std::int32_t id;
    } device;
    switch (array->buffer()->device()->platform()) {
        case Platform::Cpu:
            device.type = nanobind::device::cpu::value;
            device.id = 0;
            break;
        case Platform::Metal:
            device.type = nanobind::device::metal::value;
            device.id = std::dynamic_pointer_cast<const sx::Metal::Device>(array->buffer()->device())->id();
            break;
    }

    nanobind::dlpack::dtype_code code;
    switch (dtype) {
        case Dtype::Bool:
            code = nanobind::dlpack::dtype_code::Bool;
            break;
        case Dtype::Uint8:
            code = nanobind::dlpack::dtype_code::UInt;
            break;
        case Dtype::Int32:
            code = nanobind::dlpack::dtype_code::Int;
            break;
        case Dtype::Float16:
        case Dtype::Float32:
            code = nanobind::dlpack::dtype_code::Float;
            break;
        case Dtype::Bfloat16:
            code = nanobind::dlpack::dtype_code::Bfloat;
            break;
    }

    const nanobind::ndarray pack{
        array->buffer()->data(),
        array->shape().size(),
        array->shape().data(),
        data,
        array->strides().data(),
        {
            .code = static_cast<std::uint8_t>(code),
            .bits = static_cast<std::uint8_t>(8 * dtype_size(dtype)),
            .lanes = 1,
        },
        device.type,
        device.id
    };
    return nanobind::borrow<nanobind::capsule>(nanobind::cast(pack, nanobind::rv_policy::automatic));
}

} // namespace Dlpack

} // namespace sx
