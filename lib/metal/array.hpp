#pragma once

#include <memory>

#include "../core/array.hpp"
#include "device.hpp"

namespace sx::Metal {

[[nodiscard]] std::shared_ptr<Array<Device>>
make_array(nb::handle data, std::optional<DType> dtype, std::shared_ptr<const Device> device);

}
