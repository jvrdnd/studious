#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

#include "../buffer.hpp"

std::vector<std::int64_t> default_strides(std::span<const std::int64_t> shape, DType dtype);

struct Array {
    std::shared_ptr<Buffer> buffer;

    DType dtype;
    std::vector<std::int64_t> shape;
    std::int64_t offset_bytes;
    std::vector<std::int64_t> strides;

    Array(
        std::shared_ptr<Buffer> buffer,
        DType dtype,
        std::vector<std::int64_t> shape,
        std::int64_t offset_bytes,
        std::vector<std::int64_t> strides
    );

    Device device() const noexcept;

    std::string repr() const;
};
