#pragma once

#include <string_view>

enum class Platform { Cpu, Metal };

std::string_view platform_name(Platform platform);
