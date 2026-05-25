#pragma once

#include <cstddef>
#include <memory>
#include <span>
#include <sstream>
#include <string>

#include "device.hpp"
#include "platform.hpp"

class BaseArray {
public:
    virtual ~BaseArray() noexcept = default;

protected:
    BaseArray() = default;
};

template <typename T> class Array : public BaseArray {
public:
    [[nodiscard]] virtual std::span<const T> span() const = 0;

    [[nodiscard]] std::string repr() const {
        std::ostringstream os;
        os << "array([";

        std::span<const T> values = span();
        for (std::size_t i{ 0 }; i < values.size(); i++) {
            if (i > 0) {
                os << ", ";
            }
            os << values[i];
        }

        os << "])";
        return os.str();
    }
};

#include "../cpu/array.hpp"

template <typename T>
std::shared_ptr<Array<T>> make_array(const std::vector<float> &vector, T dtype, std::shared_ptr<const Device> device) {
    switch (device->platform()) {
        case Platform::Cpu:
            return Cpu::Array<T>{ device, vector };
        case Platform::Metal:
            break;
    }
    throw std::invalid_argument{ "invalid platform" };
}
