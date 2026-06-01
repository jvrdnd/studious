#include <algorithm>
#include <cctype>
#include <cstddef>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>
#include <string>
#include <tuple>

#include "../core/array.hpp"
#include "../core/device.hpp"
#include "../core/devices.hpp"
#include "../core/dtype.hpp"
#include "../cpu/buffer.hpp"
#include "../cpu/device.hpp"
#include "../metal/buffer.hpp"
#include "../metal/device.hpp"
#include "dlpack.hpp"

namespace sx {

NB_MODULE(_lib, m) {
    nanobind::enum_<Dtype> dtype{m, "DType"};
    for (std::size_t i{0}; i < dtype_traits.size(); ++i) {
        std::string name{dtype_traits[i].name};
        std::ranges::transform(name, name.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
        dtype.value(name.c_str(), static_cast<Dtype>(i));
    }
    dtype.export_values();

    nanobind::enum_<Platform>{m, "Platform"}
        .value("CPU", Platform::Cpu)
        .value("METAL", Platform::Metal)
        .export_values();

    nanobind::class_<Device>{m, "Device"}.def_prop_ro("platform", &Device::platform).def("__repr__", &Device::repr);
    nanobind::class_<Cpu::Device, Device> cpu_device{m, "CpuDevice"};
    nanobind::class_<Metal::Device, Device> metal_device{m, "MetalDevice"};

    m.def(
        "devices",
        [](nanobind::object platform) {
            if (platform.is_none()) {
                return devices();
            }
            return devices(nanobind::cast<Platform>(platform));
        },
        nanobind::arg("platform") = nanobind::none()
    );

    nanobind::class_<Array<Cpu::Buffer>>{m, "CpuArray"}
        .def(
            "__dlpack__",
            [](nanobind::object self, nanobind::handle) { return to_dlpack<Cpu::Buffer>(self); },
            nanobind::kw_only(),
            nanobind::arg("stream") = nanobind::none()
        )
        .def("__dlpack_device__", [](const Array<Cpu::Buffer> &) {
            return std::make_tuple(nanobind::device::cpu::value, 0);
        });

    m.def(
        "make_array",
        [](nanobind::handle data, std::optional<Dtype> dtype, std::shared_ptr<const Cpu::Device> device) {
            if (!device) {
                device = std::make_shared<Cpu::Device>();
            }
            return make_array<Cpu::Device, Cpu::Buffer>(data, dtype, std::move(device));
        },
        nanobind::arg("data"),
        nanobind::arg("dtype") = nanobind::none(),
        nanobind::kw_only(),
        nanobind::arg("device") = nullptr
    );

    nanobind::class_<Array<Metal::Buffer>>{m, "MetalArray"}
        .def(
            "__dlpack__",
            [](nanobind::object self, nanobind::handle) { return to_dlpack<Metal::Buffer>(self); },
            nanobind::kw_only(),
            nanobind::arg("stream") = nanobind::none()
        )
        .def("__dlpack_device__", [](const Array<Metal::Buffer> &self) {
            return std::make_tuple(nanobind::device::metal::value, self.buffer()->device()->id());
        });

    m.def(
        "make_array",
        [](nanobind::handle data, std::optional<Dtype> dtype, std::shared_ptr<const Metal::Device> device) {
            return make_array<Metal::Device, Metal::Buffer>(data, dtype, std::move(device));
        },
        nanobind::arg("data"),
        nanobind::arg("dtype") = nanobind::none(),
        nanobind::kw_only(),
        nanobind::arg("device")
    );

    nanobind::class_<DlpackArray> dlpack_array{m, "DlpackArray"};
    dlpack_array.def_ro("device", &DlpackArray::device)
        .def_ro("dtype", &DlpackArray::dtype)
        .def_ro("shape", &DlpackArray::shape)
        .def_ro("strides", &DlpackArray::strides)
        .def_ro("data", &DlpackArray::data);
    nanobind::class_<DlpackArray::Device>{dlpack_array, "Device"}
        .def_ro("platform", &DlpackArray::Device::platform)
        .def_ro("id", &DlpackArray::Device::id);
    m.def("from_dlpack", &from_dlpack);
}

} // namespace sx
