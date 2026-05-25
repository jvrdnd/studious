#include <nanobind/nanobind.h>
#include <vector>

#include "array.hpp"

namespace nb = nanobind;

template <typename T> std::vector<T> to_vector(nb::iterable iterable) {
    std::vector<T> vector;
    for (nb::handle item : iterable) {
        vector.push_back(nb::cast<T>(item));
    }
    return vector;
}

std::shared_ptr<::BaseArray> make_array(std::shared_ptr<const Device> device, nanobind::iterable iterable, DType dtype);
