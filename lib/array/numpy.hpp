#pragma once

#include <nanobind/ndarray.h>

#include "array.hpp"

template <typename T>
concept NPType = std::same_as<T, bool> ||
    std::same_as<T, std::int32_t> ||
    std::same_as<T, std::int64_t> ||
    std::same_as<T, float> ||
    std::same_as<T, double>;

template <NPType T> Array from_numpy(const nanobind::ndarray<nanobind::numpy, T> owner);
