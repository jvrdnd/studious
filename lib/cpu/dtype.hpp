#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "../core/dtype.hpp"

static_assert(sizeof(std::uint8_t) == dtype_size(DType::Bool)); // bool saved a uint8
static_assert(sizeof(std::uint8_t) == dtype_size(DType::UInt8));
static_assert(sizeof(std::int32_t) == dtype_size(DType::Int32));
static_assert(sizeof(float) == dtype_size(DType::Float32));

namespace Cpu {

template <typename T>
concept NativeType = std::same_as<T, std::uint8_t> ||
    std::same_as<T, std::int32_t> ||
    std::same_as<T, std::int64_t> ||
    std::same_as<T, float>;

template <DType T> consteval auto to_native_type() {
    if constexpr (T == DType::Bool) {
    }
}

template <typename F> decltype(auto) dispatch_dtype_impl(DType dtype, F &&f) {
    switch (dtype) {
        case DType::Bool:
        case DType::UInt8:
            return std::forward<F>(f).template operator()<std::uint8_t>();
        case DType::Int32:
            return std::forward<F>(f).template operator()<std::int32_t>();
        case DType::Float16:
            break;
        case DType::Float32:
            return std::forward<F>(f).template operator()<float>();
        case DType::BFloat16:
            break;
    }
    throw std::invalid_argument{"invalid cpu dtype"};
}

} // namespace Cpu
