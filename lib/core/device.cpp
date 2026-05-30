#include <string>

#include "device.hpp"
#include "platform.hpp"

namespace sx {

std::string Device::repr() const {
    return std::string(platform_name(platform())) + "(" + repr_() + ")";
}

} // namespace sx
