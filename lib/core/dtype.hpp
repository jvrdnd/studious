#pragma once

#include <array>
#include <cstddef>
#include <string_view>

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

} // namespace sx
