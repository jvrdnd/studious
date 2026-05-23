#include <nanobind/nanobind.h>
#include <stdexcept>

#include "numpy.hpp"

namespace nb = nanobind;

template <NPType T> consteval DType dtype() {
    if constexpr (std::is_same_v<T, bool>) {
        return DType::BOOL;
    } else if constexpr (std::is_same_v<T, std::int32_t>) {
        return DType::INT32;
    } else if constexpr (std::is_same_v<T, std::int64_t>) {
        return DType::INT64;
    } else if constexpr (std::is_same_v<T, float>) {
        return DType::FLOAT32;
    } else if constexpr (std::is_same_v<T, double>) {
        return DType::FLOAT64;
    }
}

template <NPType T> Array from_numpy(nb::ndarray<nb::numpy, T> owner) {
    if (owner.device_type() != nb::device::cpu::value) {
        throw std::invalid_argument("expected a CPU NumPy array");
    }

    std::vector<std::int64_t> shape;
    shape.reserve(owner.ndim());

    std::vector<std::int64_t> strides;
    strides.reserve(owner.ndim());

    for (std::size_t i{ 0 }; i < owner.ndim(); ++i) {
        shape.push_back(static_cast<std::int64_t>(owner.shape(i)));
        strides.push_back(owner.stride(i) * static_cast<std::int64_t>(sizeof(T)));
    }

    void *ptr{ static_cast<void *>(owner.data()) };
    auto buffer{ std::make_shared<NumPyBuffer>(ptr, std::move(nb::cast(owner))) };

    return Array{ std::move(buffer), dtype<T>(), std::move(shape), 0, std::move(strides) };
}

template Array from_numpy<bool>(nb::ndarray<nb::numpy, bool>);
template Array from_numpy<std::int32_t>(nb::ndarray<nb::numpy, std::int32_t>);
template Array from_numpy<std::int64_t>(nb::ndarray<nb::numpy, std::int64_t>);
template Array from_numpy<float>(nb::ndarray<nb::numpy, float>);
template Array from_numpy<double>(nb::ndarray<nb::numpy, double>);
