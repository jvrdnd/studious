#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "../core/array.hpp"
#include "../core/dtype.hpp"
#include "buffer.hpp"
#include "dtype.hpp"

namespace Cpu {

class Array final : public ::Array {
public:
    Array(std::shared_ptr<const Buffer> buffer, DType dtype, std::vector<std::size_t> shape) noexcept :
        buffer_{std::move(buffer)}, dtype_{dtype}, shape_{std::move(shape)} {};

private:
    const std::shared_ptr<const Buffer> buffer_;
    const DType dtype_;
    const std::vector<std::size_t> shape_;
};

constexpr auto dispatch_dtype = []<typename F>(DType dtype, F &&f) -> decltype(auto) {
    return dispatch_dtype_impl(dtype, std::forward<F>(f));
};

} // namespace Cpu
