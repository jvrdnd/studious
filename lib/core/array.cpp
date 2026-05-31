#include <cstddef>
#include <nanobind/nanobind.h>
#include <optional>
#include <vector>

#include "array.hpp"
#include "dtype.hpp"

namespace sx {

std::vector<std::size_t> infer_nb_shape(nanobind::handle data) {
    std::vector<std::size_t> shape;
    while (nanobind::isinstance<nanobind::list>(data)) {
        const auto list = nanobind::borrow<nanobind::list>(data);
        shape.push_back(list.size());
        if (list.size() == 0) {
            break;
        }
        data = list[0];
    }
    return shape;
}

std::vector<std::int64_t> infer_strides(const std::vector<std::size_t> &shape) {
    std::vector<std::int64_t> strides(shape.size());
    std::int64_t stride{1};
    for (std::size_t i{shape.size()}; i > 0; --i) {
        strides[i - 1] = stride;
        stride *= shape[i - 1];
    }
    return strides;
}

std::optional<DType> infer_nb_dtype(nanobind::handle data, const std::vector<std::size_t> &shape, std::size_t depth) {
    if (depth == shape.size()) {
        if (nanobind::isinstance<nanobind::bool_>(data)) {
            return DType::Bool;
        }
        if (nanobind::isinstance<nanobind::int_>(data)) {
            return DType::Int32;
        }
        if (nanobind::isinstance<nanobind::float_>(data)) {
            return DType::Float32;
        }
        throw nanobind::type_error("invalid array scalar");
    }

    if (!nanobind::isinstance<nanobind::list>(data)) {
        throw nanobind::type_error("data must be array or scalar");
    }

    const auto list{nanobind::borrow<nanobind::list>(data)};
    if (shape[depth] != list.size()) {
        throw nanobind::type_error("invalid array shape");
    }

    std::optional<DType> dtype;
    for (std::size_t i{0}; i < list.size(); i++) {
        std::optional<DType> item_dtype{infer_nb_dtype(list[i], shape, depth + 1)};
        if (dtype == DType::Float32 || item_dtype == DType::Float32) {
            dtype = DType::Float32;
        } else if (dtype == DType::Int32 || item_dtype == DType::Int32) {
            dtype = DType::Int32;
        } else if (dtype == DType::Bool || item_dtype == DType::Bool) {
            dtype = DType::Bool;
        }
    }

    return dtype;
}

} // namespace sx
