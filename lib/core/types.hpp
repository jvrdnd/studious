#pragma once

#include <cstdint>

namespace sx {

struct Bool {
    std::uint8_t bits;
};

[[nodiscard]] bool u8_to_bool(std::uint8_t x) noexcept;
[[nodiscard]] std::uint8_t bool_to_u8(bool x) noexcept;

struct Float16 {
    std::uint16_t bits;
};

[[nodiscard]] std::uint16_t f32_to_f16(float f) noexcept;
[[nodiscard]] float f16_to_f32(uint16_t h) noexcept;

struct BFloat16 {
    std::uint16_t bits;
};

[[nodiscard]] std::uint16_t f32_to_bf16(float f) noexcept;
[[nodiscard]] float bf16_to_f32(std::uint16_t h) noexcept;

} // namespace sx
