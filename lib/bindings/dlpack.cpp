#include <algorithm>
#include <cstddef>
#include <memory>
#include <nanobind/ndarray.h>
#include <stdexcept>
#include <vector>

#include "../core/device.hpp"
#include "../core/platform.hpp"
#include "../cpu/device.hpp"
#include "../cpu/read.hpp"
#include "../metal/device.hpp"
#include "../metal/devices.hpp"
#include "dlpack.hpp"

namespace sx::Dlpack {

namespace {
nanobind::ndarray<> get_array(nanobind::object data) {
    const auto array = nanobind::cast<nanobind::ndarray<>>(data);
    if (!array.is_valid()) {
        throw std::runtime_error{"invalid ndarray"};
    }
    return array;
}
} // namespace

DevicePtr get_device(nanobind::object data) {
    const nanobind::ndarray<> array{get_array(data)};
    switch (array.device_type()) {
        case nanobind::device::cpu::value:
            return std::make_shared<sx::Cpu::Device>();
        case nanobind::device::metal::value:
            std::vector<DevicePtr> devices{sx::Metal::devices()};
            auto it{std::find_if(devices.begin(), devices.end(), [array](const DevicePtr &device) {
                return std::dynamic_pointer_cast<const sx::Metal::Device>(device)->id() == array.device_id();
            })};
            if (it == devices.end()) {
                throw std::logic_error{"invalid metal device id"};
            }
            return *it;
    }
    throw std::invalid_argument{"invalid device type"};
}

Dtype get_dtype(nanobind::object data) {
    const nanobind::dlpack::dtype dtype{get_array(data).dtype()};
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

std::vector<std::size_t> get_shape(nanobind::object data) {
    const nanobind::ndarray<> array{get_array(data)};
    std::vector<std::size_t> shape;
    shape.reserve(array.ndim());
    for (std::size_t i{0}; i < array.ndim(); ++i) {
        shape.push_back(array.shape(i));
    }
    return shape;
}

std::vector<std::ptrdiff_t> get_strides(nanobind::object data) {
    const nanobind::ndarray<> array{get_array(data)};
    std::vector<std::ptrdiff_t> strides;
    strides.reserve(array.ndim());
    for (std::size_t i{0}; i < array.ndim(); ++i) {
        strides.push_back(array.stride(i));
    }
    return strides;
}

std::variant<std::vector<bool>, std::vector<std::int32_t>, std::vector<float>> get_data(nanobind::object data) {
    const DevicePtr device{get_device(data)};
    const Dtype dtype{get_dtype(data)};
    const std::vector<std::size_t> shape{get_shape(data)};
    const std::vector<std::ptrdiff_t> strides{get_strides(data)};

    const auto bytes = static_cast<std::byte *>(get_array(data).data());
    switch (device->platform()) {
        case Platform::Cpu:
        case Platform::Metal: {
            switch (dtype) {
                case Dtype::Bool:
                    return read_buffer<bool>(bytes, dtype, shape, strides);
                case Dtype::Uint8:
                case Dtype::Int32:
                    return read_buffer<int32_t>(bytes, dtype, shape, strides);
                case Dtype::Float16:
                case Dtype::Float32:
                case Dtype::Bfloat16:
                    return read_buffer<float>(bytes, dtype, shape, strides);
            }
        }
    }

    throw std::invalid_argument{"invalid platform"};
}

} // namespace sx::Dlpack
