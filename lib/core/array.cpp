#include <cstddef>
#include <nanobind/nanobind.h>
#include <optional>
#include <vector>

#include "array.hpp"
#include "dtype.hpp"

namespace sx {

namespace nb = nanobind;

std::vector<std::size_t> infer_nb_shape(nb::handle data) {
    std::vector<std::size_t> shape;
    while (nb::isinstance<nb::list>(data)) {
        auto list = nb::borrow<nb::list>(data);
        shape.push_back(list.size());
        if (list.size() == 0) {
            break;
        }
        data = list[0];
    }
    return shape;
}

std::optional<DType> infer_nb_dtype(nb::handle data, const std::vector<std::size_t> &shape, std::size_t depth) {
    if (depth == shape.size()) {
        if (nb::isinstance<nb::bool_>(data)) {
            return DType::Bool;
        }
        if (nb::isinstance<nb::int_>(data)) {
            return DType::Int32;
        }
        if (nb::isinstance<nb::float_>(data)) {
            return DType::Float32;
        }
        throw nb::type_error("invalid array scalar");
    }

    if (!nb::isinstance<nb::list>(data)) {
        throw nb::type_error("data must be array or scalar");
    }

    auto list{nb::borrow<nb::list>(data)};
    if (shape[depth] != list.size()) {
        throw nb::type_error("invalid array shape");
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
