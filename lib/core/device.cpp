#include <string>

#include "device.hpp"
#include "platform.hpp"

std::string Device::repr() const {
    return std::string(platform_name(platform())) + "(" + repr_() + ")";
}
