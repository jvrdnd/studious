#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <sstream>
#include <utility>
#include <vector>

#include "../buffer.hpp"
#include "array.hpp"

std::vector<std::int64_t> default_strides(std::span<const std::int64_t> shape, DType dtype) {
    std::vector<std::int64_t> strides(shape.size());
    std::int64_t stride{ static_cast<std::int64_t>(dtype_size(dtype)) };

    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

Array::Array(
    std::shared_ptr<Buffer> buffer,
    DType dtype,
    std::vector<std::int64_t> shape,
    std::int64_t offset_bytes,
    std::vector<std::int64_t> strides
) :
    buffer{ std::move(buffer) },
    dtype{ dtype },
    shape{ std::move(shape) },
    offset_bytes{ offset_bytes },
    strides{ std::move(strides) } {}

Device Array::device() const noexcept {
    return buffer->device;
}

std::string Array::repr() const {
    return "test";
}
