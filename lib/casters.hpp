#include <nanobind/nanobind.h>

#include "core/types.hpp"

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <> struct type_caster<sx::Bool> {
    NB_TYPE_CASTER(sx::Bool, const_name("Bool"))

    bool from_python(handle src, std::uint8_t, cleanup_list *) noexcept {
        try {
            value.bits = sx::bool_to_u8(nanobind::cast<bool>(src));
            return true;
        } catch (...) {
            return false;
        }
    }

    static handle from_cpp(sx::Bool value, rv_policy, cleanup_list *) noexcept {
        try {
            return nanobind::cast(sx::u8_to_bool(value.bits)).release();
        } catch (...) {
            return {};
        }
    }
};

template <> struct type_caster<sx::Float16> {
    NB_TYPE_CASTER(sx::Float16, const_name("Float16"))

    bool from_python(handle src, std::uint16_t, cleanup_list *) noexcept {
        try {
            value.bits = sx::f32_to_f16(nanobind::cast<float>(src));
            return true;
        } catch (...) {
            return false;
        }
    }

    static handle from_cpp(sx::Float16 value, rv_policy, cleanup_list *) noexcept {
        try {
            return nanobind::cast(sx::f16_to_f32(value.bits)).release();
        } catch (...) {
            return {};
        }
    }
};

template <> struct type_caster<sx::BFloat16> {
    NB_TYPE_CASTER(sx::BFloat16, const_name("BFloat16"))

    bool from_python(handle src, std::uint16_t, cleanup_list *) noexcept {
        try {
            value.bits = sx::f32_to_bf16(nanobind::cast<float>(src));
            return true;
        } catch (...) {
            return false;
        }
    }

    static handle from_cpp(sx::BFloat16 value, rv_policy, cleanup_list *) noexcept {
        try {
            return nanobind::cast(sx::bf16_to_f32(value.bits)).release();
        } catch (...) {
            return {};
        }
    }
};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
