#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "./core/array.hpp"
#include "./core/device.hpp"
#include "./core/devices.hpp"
#include "./cpu/array.hpp"
#include "./cpu/device.hpp"

namespace nb = nanobind;

NB_MODULE(_lib, m) {
    nb::enum_<Platform>{ m, "Platform" }.value("CPU", Platform::Cpu).value("METAL", Platform::Metal).export_values();

    nb::class_<Device>{ m, "Device" }.def_prop_ro("platform", &Device::platform).def("__repr__", &Device::repr);
    nb::class_<Cpu::Device, Device> CpuDevice_{ m, "CpuDevice" };

    // devices(platform: Platform | None = None)
    m.def(
        "devices",
        [](nb::object platform) {
            if (platform.is_none()) {
                return devices();
            }
            return devices(nb::cast<Platform>(platform));
        },
        nb::arg("platform") = nb::none()
    );

    nb::class_<::Array<float>> Array_{ m, "Array" };
    nb::class_<Cpu::Array<float>, ::Array<float>>{ m, "CpuArray" }
        .def(
            nb::init<std::shared_ptr<const Cpu::Device>, const std::vector<float> &>(),
            nb::arg("device"),
            nb::arg("values")
        )
        .def("__repr__", &Cpu::Array<float>::repr);
}
