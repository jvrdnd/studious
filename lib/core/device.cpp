#include <string>

#include "device.hpp"

std::string Device::repr() const {
    std::string repr;

    switch (platform()) {
        case Platform::Cpu:
            repr = "cpu";
            break;
        case Platform::Metal:
            repr = "metal";
            break;
    }

    repr += "(" + repr_() + ")";

    return repr;
}
