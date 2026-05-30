#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "types.hpp"

namespace sx {

enum class DType { Bool, UInt8, Int32, Float16, Float32, BFloat16 };

namespace {

struct dtype_trait {
    std::string_view name;
    std::size_t size;
};
inline constexpr std::array<dtype_trait, 7> dtype_traits = {
    dtype_trait{"pred", 1},
    dtype_trait{"u8", 1},
    dtype_trait{"s32", 4},
    dtype_trait{"f16", 2},
    dtype_trait{"f32", 4},
    dtype_trait{"bf16", 2},
};

} // namespace

constexpr std::string_view dtype_name(DType dtype) {
    return dtype_traits[static_cast<std::size_t>(dtype)].name;
}
constexpr std::size_t dtype_size(DType dtype) {
    return dtype_traits[static_cast<std::size_t>(dtype)].size;
}

template <typename F> decltype(auto) dispatch_dtype_impl(DType dtype, F &&f) {
    switch (dtype) {
        case DType::Bool:
            return std::forward<F>(f).template operator()<Bool>();
        case DType::UInt8:
            return std::forward<F>(f).template operator()<std::uint8_t>();
        case DType::Int32:
            return std::forward<F>(f).template operator()<std::int32_t>();
        case DType::Float16:
            return std::forward<F>(f).template operator()<Float16>();
        case DType::Float32:
            return std::forward<F>(f).template operator()<float>();
        case DType::BFloat16:
            return std::forward<F>(f).template operator()<BFloat16>();
    }
}

inline auto dispatch_dtype = [](DType dtype, auto &&f) -> decltype(auto) {
    return dispatch_dtype_impl(dtype, std::forward<decltype(f)>(f));
};

} // namespace sx
