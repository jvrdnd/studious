#include <nanobind/ndarray.h>
#include <stdexcept>

#include "../cpu/read.hpp"
#include "dlpack.hpp"

namespace sx {

Platform from_dlpack_platform(std::int32_t platform) {
    switch (platform) {
        case nanobind::device::cpu::value:
            return Platform::Cpu;
        case nanobind::device::metal::value:
            return Platform::Metal;
        default:
            break;
    }
    throw std::invalid_argument{"invalid dlpack platform"};
}

Dtype from_dlpack_dtype(nanobind::dlpack::dtype dtype) {
    if (dtype.lanes == 1) {
        switch (static_cast<nanobind::dlpack::dtype_code>(dtype.code)) {
            case nanobind::dlpack::dtype_code::Bool:
                return Dtype::Bool;
            case nanobind::dlpack::dtype_code::UInt:
                if (dtype.bits == 8) {
                    return Dtype::Uint8;
                }
            case nanobind::dlpack::dtype_code::Int:
                if (dtype.bits == 32) {
                    return Dtype::Int32;
                }
            case nanobind::dlpack::dtype_code::Float:
                if (dtype.bits == 16) {
                    return Dtype::Float16;
                }
                if (dtype.bits == 32) {
                    return Dtype::Float32;
                }
            case nanobind::dlpack::dtype_code::Bfloat:
                if (dtype.bits == 16) {
                    return Dtype::Bfloat16;
                }
            default:
                break;
        }
    }
    throw std::invalid_argument{"invalid dlpack dtype"};
}

DlpackArray from_dlpack(nanobind::object data) {
    const auto array = nanobind::cast<nanobind::ndarray<>>(data);
    if (!array.is_valid()) {
        throw std::runtime_error{"invalid ndarray"};
    }

    DlpackArray view;

    view.device.platform = from_dlpack_platform(array.device_type());
    view.device.id = array.device_id();

    view.dtype = from_dlpack_dtype(array.dtype());

    const std::size_t ndim{array.ndim()};
    view.shape.reserve(ndim);
    view.strides.reserve(ndim);
    for (std::size_t i{0}; i < ndim; ++i) {
        view.shape.push_back(array.shape(i));
        view.strides.push_back(array.stride(i));
    }

    const auto bytes = static_cast<std::byte *>(array.data());
    switch (view.dtype) {
        case Dtype::Bool:
            view.data = read_buffer<bool>(bytes, view.dtype, view.shape, view.strides);
            break;
        case Dtype::Uint8:
        case Dtype::Int32:
            view.data = read_buffer<int32_t>(bytes, view.dtype, view.shape, view.strides);
            break;
        case Dtype::Float16:
        case Dtype::Float32:
        case Dtype::Bfloat16:
            view.data = read_buffer<float>(bytes, view.dtype, view.shape, view.strides);
            break;
    }

    return view;
}

nanobind::dlpack::dtype_code to_dlpack_dtype_code(Dtype dtype) {
    switch (dtype) {
        case Dtype::Bool:
            return nanobind::dlpack::dtype_code::Bool;
        case Dtype::Uint8:
            return nanobind::dlpack::dtype_code::UInt;
        case Dtype::Int32:
            return nanobind::dlpack::dtype_code::Int;
        case Dtype::Float16:
        case Dtype::Float32:
            return nanobind::dlpack::dtype_code::Float;
        case Dtype::Bfloat16:
            return nanobind::dlpack::dtype_code::Bfloat;
    }
}

} // namespace sx
