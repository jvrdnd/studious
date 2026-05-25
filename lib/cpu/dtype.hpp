#pragma once

#include <cstddef>
#include <cstdint>

#include "../core/dtype.hpp"

static_assert(sizeof(bool) == dtype_size(DType::Bool));
static_assert(sizeof(std::uint8_t) == dtype_size(DType::UInt8));
static_assert(sizeof(std::int32_t) == dtype_size(DType::Int32));
static_assert(sizeof(std::int64_t) == dtype_size(DType::Int64));
static_assert(sizeof(float) == dtype_size(DType::Float32));

namespace Cpu {

template <typename T>
concept NativeType = std::same_as<T, bool> ||
    std::same_as<T, std::uint8_t> ||
    std::same_as<T, std::int32_t> ||
    std::same_as<T, std::int64_t> ||
    std::same_as<T, float>;

template <NativeType T> consteval DType to_dtype() {
    if constexpr (std::same_as<T, bool>) {
        return DType::Bool;
    } else if constexpr (std::same_as<T, std::uint8_t>) {
        return DType::UInt8;
    } else if constexpr (std::same_as<T, std::int32_t>) {
        return DType::Int32;
    } else if constexpr (std::same_as<T, std::int64_t>) {
        return DType::Int64;
    } else if constexpr (std::same_as<T, float>) {
        return DType::Float32;
    }
}

template <class F> decltype(auto) dispatch_dtype(DType dtype, F &&f) {
    switch (dtype) {
        case DType::Bool:
        case DType::UInt8:
            return std::forward<F>(f).template operator()<std::uint8_t>();
        case DType::Int32:
            return std::forward<F>(f).template operator()<std::int32_t>();
        case DType::Int64:
            return std::forward<F>(f).template operator()<std::int64_t>();
        case DType::Float32:
            return std::forward<F>(f).template operator()<float>();
        default:
            break;
    }
    throw std::invalid_argument{ "invalid cpu dtype" };
}

} // namespace Cpu
