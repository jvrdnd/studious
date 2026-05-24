#include <string>

#include "device.hpp"

std::string Device::repr() const {
    std::string repr;

    switch (platform()) {
        case Platform::CPU:
            repr = "cpu";
            break;
        case Platform::METAL:
            repr = "metal";
            break;
    }

    repr += "(" + repr_() + ")";

    return repr;
}
