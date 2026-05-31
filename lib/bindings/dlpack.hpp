#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "../core/array.hpp"
#include "../core/dtype.hpp"

namespace sx {

[[nodiscard]] constexpr nanobind::dlpack::dtype_code dtype_code(DType dtype) {
    using dtype_code = nanobind::dlpack::dtype_code;
    switch (dtype) {
        case DType::Bool:
            return dtype_code::Bool;
        case DType::UInt8:
            return dtype_code::UInt;
        case DType::Int32:
            return dtype_code::Int;
        case DType::Float16:
        case DType::Float32:
            return dtype_code::Float;
        case DType::BFloat16:
            return dtype_code::Bfloat;
    }
}

template <BufferType B> nanobind::capsule dlpack(nanobind::handle data) {
    const auto *array = nanobind::cast<Array<B> *>(data);
    const DType dtype = array->dtype();
    const nanobind::ndarray pack{
        array->buffer()->data(),
        array->shape().size(),
        array->shape().data(),
        data,
        array->strides().data(),
        {
            .code = static_cast<std::uint8_t>(dtype_code(dtype)),
            .bits = static_cast<std::uint8_t>(8 * dtype_size(dtype)),
            .lanes = 1,
        }
    };
    return nanobind::borrow<nanobind::capsule>(nanobind::cast(pack, nanobind::rv_policy::automatic));
}

} // namespace sx
