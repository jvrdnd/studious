#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "./core/device.hpp"
#include "./core/devices.hpp"

namespace nb = nanobind;

NB_MODULE(_lib, m) {
    nb::enum_<Platform>(m, "Platform").value("CPU", Platform::CPU).value("METAL", Platform::METAL).export_values();

    nb::class_<Device>(m, "Device").def_prop_ro("platform", &Device::platform).def("__repr__", &Device::repr);

    m.def("devices", nb::overload_cast<>(&devices), "Return all available devices.");
    m.def(
        "devices", nb::overload_cast<Platform>(&devices), "Return devices for a specific platform.", nb::arg("platform")
    );
}
