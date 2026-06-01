#pragma once

#include <concepts>
#include <cstdint>
#include <ostream>

namespace sx {

struct Bool {
    std::uint8_t bits;
};
std::ostream &operator<<(std::ostream &os, Bool b);

[[nodiscard]] bool u8_to_bool(std::uint8_t u) noexcept;
[[nodiscard]] std::uint8_t bool_to_u8(bool b) noexcept;

struct Float16 {
    std::uint16_t bits;
};
std::ostream &operator<<(std::ostream &os, Float16 h);

[[nodiscard]] std::uint16_t f32_to_f16(float f) noexcept;
[[nodiscard]] float f16_to_f32(uint16_t h) noexcept;

struct Bfloat16 {
    std::uint16_t bits;
};
std::ostream &operator<<(std::ostream &os, Bfloat16 h);

[[nodiscard]] std::uint16_t f32_to_bf16(float f) noexcept;
[[nodiscard]] float bf16_to_f32(std::uint16_t h) noexcept;

template <typename T> [[nodiscard]] T cast(auto x) {
    if constexpr (std::same_as<T, Bool>) {
        return Bool{bool_to_u8(static_cast<bool>(x))};
    } else if constexpr (std::same_as<T, Float16>) {
        return Float16{f32_to_f16(static_cast<float>(x))};
    } else if constexpr (std::same_as<T, Bfloat16>) {
        return Bfloat16{f32_to_bf16(static_cast<float>(x))};
    } else {
        return static_cast<T>(x);
    }
}

} // namespace sx
