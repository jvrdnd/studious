#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "../../core/dtype.hpp"
#include "../../core/types.hpp"
#include "iterate.hpp"

namespace sx::Cpu {

template <typename T>
std::vector<T>
vectorize(std::byte *data, Dtype dtype, std::span<const std::size_t> shape, std::span<const std::ptrdiff_t> strides) {
    std::vector<T> xs;
    dispatch_dtype(dtype, [&]<typename U>() {
        iterate(reinterpret_cast<U *>(data), shape, strides, [&](U x) { xs.push_back(static_cast_to_native<T>(x)); });
    });
    return xs;
}

} // namespace sx::Cpu
