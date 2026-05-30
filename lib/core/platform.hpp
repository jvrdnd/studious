#pragma once

#include <string_view>

namespace sx {

enum class Platform { Cpu, Metal };

std::string_view platform_name(Platform platform);

} // namespace sx
