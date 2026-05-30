#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "casters.hpp"
#include "core/array.hpp"
#include "core/device.hpp"
#include "core/devices.hpp"
#include "core/dtype.hpp"
#include "cpu/buffer.hpp"
#include "cpu/device.hpp"
#include "metal/buffer.hpp"
#include "metal/device.hpp"

namespace sx {

namespace nb = nanobind;

NB_MODULE(_lib, m) {
    nb::enum_<DType>(m, "DType")
        .value("PRED", DType::Bool)
        .value("U8", DType::UInt8)
        .value("S32", DType::Int32)
        .value("F16", DType::Float16)
        .value("F32", DType::Float32)
        .value("BF16", DType::BFloat16);

    nb::enum_<Platform>{m, "Platform"}.value("CPU", Platform::Cpu).value("METAL", Platform::Metal).export_values();

    nb::class_<Device>{m, "Device"}.def_prop_ro("platform", &Device::platform).def("__repr__", &Device::repr);
    nb::class_<Cpu::Device, Device> CpuDevice_{m, "CpuDevice"};
    nb::class_<Metal::Device, Device> MetalDevice_{m, "MetalDevice"};

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

    nb::class_<Array<Cpu::Buffer>> CpuArray_{m, "CpuArray"};
    m.def(
        "make_array",
        [](nb::handle data, std::optional<DType> dtype, std::shared_ptr<const Cpu::Device> device) {
            if (!device) {
                device = std::make_shared<Cpu::Device>();
            }
            return make_array<Cpu::Device, Cpu::Buffer>(data, dtype, std::move(device));
        },
        nb::arg("data"),
        nb::arg("dtype") = nb::none(),
        nb::arg("device") = nullptr
    );

    nb::class_<Array<Metal::Buffer>> MetalArray_{m, "MetalArray"};
    m.def(
        "make_array",
        [](nb::handle data, std::optional<DType> dtype, std::shared_ptr<const Metal::Device> device) {
            return make_array<Metal::Device, Metal::Buffer>(data, dtype, std::move(device));
        },
        nb::arg("data"),
        nb::arg("dtype") = nb::none(),
        nb::arg("device")
    );
}

} // namespace sx
