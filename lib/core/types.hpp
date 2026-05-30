#pragma once

#include <cstdint>

namespace sx {

struct Bool {
    std::uint8_t bits;
    constexpr Bool() : bits(0) {}
    constexpr Bool(bool x) : bits(static_cast<std::uint8_t>(x)) {}
    constexpr explicit Bool(std::uint8_t x) : bits(static_cast<std::uint8_t>(x != 0)) {}
    constexpr explicit operator bool() const {
        return bits != 0;
    }
};

[[nodiscard]] bool u8_to_bool(std::uint8_t x);
[[nodiscard]] std::uint8_t bool_to_u8(bool x);

struct Float16 {
    std::uint16_t bits;
};

[[nodiscard]] std::uint16_t f32_to_f16(float f);
[[nodiscard]] float f16_to_f32(uint16_t h);

struct BFloat16 {
    std::uint16_t bits;
};

[[nodiscard]] std::uint16_t f32_to_bf16(float f);
[[nodiscard]] float bf16_to_f32(std::uint16_t h);

} // namespace sx
