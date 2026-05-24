#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <optional>

#include "./core/device.hpp"
#include "./core/devices.hpp"

namespace nb = nanobind;

NB_MODULE(_lib, m) {
    nb::class_<Device>(m, "Device").def_prop_ro("platform", &Device::platform).def("__repr__", &Device::repr);

    m.def(
        "devices",
        [](std::optional<Platform> platform) {
            if (platform.has_value()) {
                return devices(*platform);
            }
            return devices();
        },
        nb::arg("platform") = nb::none(),
        "Return available devices, optionally filtered by platform."
    );
}
