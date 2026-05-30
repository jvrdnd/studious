#pragma once

#include <cstdint>

#include "../core/dtype.hpp"

namespace sx::Cpu {

static_assert(sizeof(Bool) == dtype_size(DType::Bool));
static_assert(sizeof(std::uint8_t) == dtype_size(DType::UInt8));
static_assert(sizeof(std::int32_t) == dtype_size(DType::Int32));
static_assert(sizeof(float) == dtype_size(DType::Float32));

} // namespace sx::Cpu
