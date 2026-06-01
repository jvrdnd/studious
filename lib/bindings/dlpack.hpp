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

namespace sx {

struct DlpackArray {
    struct Device {
        Platform platform;
        std::int32_t id;
    };

    Device device;
    Dtype dtype;
    std::vector<std::size_t> shape;
    std::vector<std::ptrdiff_t> strides;
    std::variant<std::vector<bool>, std::vector<std::int32_t>, std::vector<float>> data;
};
[[nodiscard]] Platform from_dlpack_platform(std::int32_t platform);
[[nodiscard]] Dtype from_dlpack_dtype(nanobind::dlpack::dtype dtype);
[[nodiscard]] DlpackArray from_dlpack(nanobind::object data);

[[nodiscard]] nanobind::dlpack::dtype_code to_dlpack_dtype_code(Dtype dtype);
template <BufferType B> nanobind::capsule to_dlpack(nanobind::handle data) {
    const auto *array = nanobind::cast<Array<B> *>(data);
    const Dtype dtype = array->dtype();
    const nanobind::ndarray pack{
        array->buffer()->data(),
        array->shape().size(),
        array->shape().data(),
        data,
        array->strides().data(),
        {
            .code = static_cast<std::uint8_t>(to_dlpack_dtype_code(dtype)),
            .bits = static_cast<std::uint8_t>(8 * dtype_size(dtype)),
            .lanes = 1,
        }
    };
    return nanobind::borrow<nanobind::capsule>(nanobind::cast(pack, nanobind::rv_policy::automatic));
}

} // namespace sx
