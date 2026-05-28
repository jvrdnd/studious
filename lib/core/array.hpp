#pragma once

#include <memory>
#include <nanobind/nanobind.h>
#include <span>
#include <sstream>
#include <string>

#include "device.hpp"
#include "dtype.hpp"

class Array {
public:
    virtual ~Array() noexcept = default;

protected:
    Array() = default;
};

std::shared_ptr<Array>
make_array(nanobind::handle data, std::optional<DType> dtype, std::shared_ptr<const Device> device);

template <typename T> std::string repr(std::span<const T> span) {
    std::ostringstream os;
    os << "array([";

    for (std::size_t i{0}; i < span.size(); i++) {
        if (i > 0) {
            os << ", ";
        }
        os << span[i];
    }

    os << "])";
    return os.str();
}
