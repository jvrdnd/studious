#pragma once

#include <cstddef>
#include <functional>
#include <span>
#include <vector>

namespace sx::Cpu {

template <typename T, typename F>
void iterate(const T *data, std::span<const std::size_t> shape, std::span<const std::ptrdiff_t> strides, F &&f) {
    const std::size_t ndim{shape.size()};

    if (ndim == 0) {
        std::invoke(f, *data);
        return;
    }

    std::vector<std::ptrdiff_t> reset_strides(ndim, 0);
    std::size_t size{1};
    for (std::size_t dim{0}; dim < ndim; ++dim) {
        if (shape[dim] == 0) {
            return;
        }
        reset_strides[dim] = static_cast<std::ptrdiff_t>(shape[dim]) * strides[dim];
        size *= shape[dim];
    }

    std::vector<std::size_t> indices(ndim, 0);
    std::ptrdiff_t offset{0};

    for (std::size_t item{0}; item < size; ++item) {
        std::invoke(f, data[offset]);

        std::size_t dim{ndim};
        while (dim-- > 0) {
            offset += strides[dim];

            if (++indices[dim] < shape[dim]) {
                break;
            }

            indices[dim] = 0;
            offset -= reset_strides[dim];
        }
    }
}

} // namespace sx::Cpu
