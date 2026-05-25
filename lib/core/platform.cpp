#include <stdexcept>

#include "platform.hpp"

std::string_view platform_name(Platform platform) {
    switch (platform) {
        case Platform::Cpu:
            return "cpu";
        case Platform::Metal:
            return "metal";
    }
    throw std::invalid_argument{ "invalid platform" };
}
