#include "dtype.hpp"

constexpr std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::Bool:
            return 1;
        case DType::Int32:
            return 4;
        case DType::Int64:
            return 8;
        case DType::UInt8:
            return 1;
        case DType::Float16:
            return 2;
        case DType::Float32:
            return 4;
        case DType::BFloat16:
            return 2;
    }
    throw std::runtime_error("invalid dtype");
}

constexpr std::string_view dtype_name(DType dtype) {
    switch (dtype) {
        case DType::Bool:
            return "pred";
        case DType::Int32:
            return "s32";
        case DType::Int64:
            return "s64";
        case DType::UInt8:
            return "u8";
        case DType::Float16:
            return "f16";
        case DType::Float32:
            return "f32";
        case DType::BFloat16:
            return "bf16";
    }
    throw std::runtime_error("invalid dtype");
}
