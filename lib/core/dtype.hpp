#pragma once

#include <array>
#include <cstddef>
#include <string_view>

#include "types.hpp"

namespace sx {

enum class Dtype { Bool, Uint8, Int32, Float16, Float32, Bfloat16 };

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

[[nodiscard]] constexpr std::string_view dtype_name(Dtype dtype) noexcept {
    return dtype_traits[static_cast<std::size_t>(dtype)].name;
}
[[nodiscard]] constexpr std::size_t dtype_size(Dtype dtype) noexcept {
    return dtype_traits[static_cast<std::size_t>(dtype)].size;
}

static_assert(sizeof(Bool) == dtype_size(Dtype::Bool));
static_assert(sizeof(std::uint8_t) == dtype_size(Dtype::Uint8));
static_assert(sizeof(std::int32_t) == dtype_size(Dtype::Int32));
static_assert(sizeof(Float16) == dtype_size(Dtype::Float16));
static_assert(sizeof(float) == dtype_size(Dtype::Float32));
static_assert(sizeof(Bfloat16) == dtype_size(Dtype::Bfloat16));

template <typename F> decltype(auto) dispatch_dtype(Dtype dtype, F &&f) {
    switch (dtype) {
        case Dtype::Bool:
            return std::forward<F>(f).template operator()<Bool>();
        case Dtype::Uint8:
            return std::forward<F>(f).template operator()<std::uint8_t>();
        case Dtype::Int32:
            return std::forward<F>(f).template operator()<std::int32_t>();
        case Dtype::Float16:
            return std::forward<F>(f).template operator()<Float16>();
        case Dtype::Float32:
            return std::forward<F>(f).template operator()<float>();
        case Dtype::Bfloat16:
            return std::forward<F>(f).template operator()<Bfloat16>();
    }
}

} // namespace sx
