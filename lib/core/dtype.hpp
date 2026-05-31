#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "types.hpp"

namespace sx {

enum class DType { Bool, UInt8, Int32, Float16, Float32, BFloat16 };

struct dtype_trait {
    std::string_view name;
    std::size_t size;
};
inline constexpr std::array<dtype_trait, 6> dtype_traits = {
    dtype_trait{"pred", 1},
    dtype_trait{"u8", 1},
    dtype_trait{"s32", 4},
    dtype_trait{"f16", 2},
    dtype_trait{"f32", 4},
    dtype_trait{"bf16", 2},
};

[[nodiscard]] constexpr std::size_t dtype_size(DType dtype) noexcept {
    return dtype_traits[static_cast<std::size_t>(dtype)].size;
}
static_assert(sizeof(Bool) == dtype_size(DType::Bool));
static_assert(sizeof(std::uint8_t) == dtype_size(DType::UInt8));
static_assert(sizeof(std::int32_t) == dtype_size(DType::Int32));
static_assert(sizeof(Float16) == dtype_size(DType::Float16));
static_assert(sizeof(float) == dtype_size(DType::Float32));
static_assert(sizeof(BFloat16) == dtype_size(DType::BFloat16));

template <typename F> decltype(auto) dispatch_dtype(DType dtype, F &&f) {
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

} // namespace sx
