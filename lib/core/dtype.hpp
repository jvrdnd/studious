#pragma once

#include <cstddef>
#include <nanobind/nanobind.h>
#include <ranges>
#include <stdexcept>
#include <string_view>

enum class DType { Bool, UInt8, Int32, Int64, Float16, Float32, BFloat16 };

constexpr std::string_view dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Bool:
            return "pred";
        case DType::UInt8:
            return "u8";
        case DType::Int32:
            return "s32";
        case DType::Int64:
            return "s64";
        case DType::Float16:
            return "f32";
        case DType::Float32:
            return "f32";
        case DType::BFloat16:
            return "f32";
    }
    throw std::runtime_error{ "invalid dtype" };
}

constexpr std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Bool:
            return 1;
        case DType::UInt8:
            return 1;
        case DType::Int32:
            return 4;
        case DType::Int64:
            return 8;
        case DType::Float16:
            return 2;
        case DType::Float32:
            return 4;
        case DType::BFloat16:
            return 2;
    }
    throw std::runtime_error{ "invalid dtype" };
}

template <typename R>
concept Range = std::ranges::contiguous_range<R> && std::ranges::sized_range<R>;
