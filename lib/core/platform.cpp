#include "platform.hpp"

namespace sx {

std::string_view platform_name(Platform platform) {
    switch (platform) {
        case Platform::Cpu:
            return "cpu";
        case Platform::Metal:
            return "metal";
    }
}

} // namespace sx
