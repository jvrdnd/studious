#include <bit>

#include "types.hpp"

namespace sx {

std::ostream &operator<<(std::ostream &os, Bool b) {
    return os << (u8_to_bool(b.bits) ? "True" : "False");
}

bool u8_to_bool(std::uint8_t u) noexcept {
    return u != 0;
}
std::uint8_t bool_to_u8(bool b) noexcept {
    return static_cast<std::uint8_t>(b);
}

std::ostream &operator<<(std::ostream &os, Float16 h) {
    return os << f16_to_f32(h.bits);
}

std::uint16_t f32_to_f16(float f) noexcept {
    std::uint32_t x = std::bit_cast<std::uint32_t>(f);

    std::uint32_t sign = (x >> 16) & 0x8000u;
    std::uint32_t mant = x & 0x007fffffu;
    int exp = int((x >> 23) & 0xffu) - 112;

    if (((x >> 23) & 0xffu) == 0xffu) {
        if (!mant) {
            return std::uint16_t(sign | 0x7c00u);
        }
        mant >>= 13;
        return std::uint16_t(sign | 0x7c00u | mant | !mant);
    }

    if (exp >= 31) {
        return std::uint16_t(sign | 0x7c00u);
    }

    if (exp <= 0) {
        if (exp < -10) {
            return std::uint16_t(sign);
        }

        mant |= 0x00800000u;
        int shift = 14 - exp;
        std::uint32_t hm = mant >> shift;
        std::uint32_t rb = 1u << (shift - 1);
        std::uint32_t rem = mant & (rb - 1);

        if ((mant & rb) && (rem || (hm & 1))) {
            ++hm;
        }

        return std::uint16_t(sign | hm);
    }

    std::uint32_t he = std::uint32_t(exp) << 10;
    std::uint32_t hm = mant >> 13;
    std::uint32_t r = mant & 0x1fffu;

    if (r > 0x1000u || (r == 0x1000u && (hm & 1))) {
        ++hm;
        if (hm == 0x400u) {
            hm = 0;
            he += 0x0400u;
            if (he >= 0x7c00u) {
                he = 0x7c00u;
            }
        }
    }

    return std::uint16_t(sign | he | hm);
}

float f16_to_f32(std::uint16_t h) noexcept {
    std::uint32_t sign = std::uint32_t(h & 0x8000u) << 16;
    std::uint32_t exp = (h >> 10) & 0x1fu;
    std::uint32_t mant = h & 0x03ffu;
    std::uint32_t out;

    if (!exp) {
        if (!mant) {
            out = sign;
        } else {
            exp = 1;
            while (!(mant & 0x0400u)) {
                mant <<= 1;
                --exp;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 112) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
        if (mant) {
            out |= 0x00400000u;
        }
    } else {
        out = sign | ((exp + 112) << 23) | (mant << 13);
    }

    return std::bit_cast<float>(out);
}

std::ostream &operator<<(std::ostream &os, BFloat16 h) {
    return os << bf16_to_f32(h.bits);
}

std::uint16_t f32_to_bf16(float f) noexcept {
    std::uint32_t x = std::bit_cast<std::uint32_t>(f);
    std::uint32_t lsb = (x >> 16) & 1u;
    std::uint32_t bias = 0x7fffu + lsb;
    return std::uint16_t((x + bias) >> 16);
}

float bf16_to_f32(std::uint16_t h) noexcept {
    std::uint32_t x = std::uint32_t(h) << 16;
    return std::bit_cast<float>(x);
}

} // namespace sx
