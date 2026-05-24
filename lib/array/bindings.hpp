#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "array.hpp"
#include "numpy.hpp"

namespace nb = nanobind;

NB_MODULE(_lib, m) {
    nb::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .value("METAL", Device::METAL)
        .export_values();

    nb::enum_<DType>(m, "DType")
        .value("BOOL", DType::BOOL)
        .value("INT32", DType::INT32)
        .value("INT64", DType::INT64)
        .value("BFLOAT16", DType::BFLOAT16)
        .value("FLOAT32", DType::FLOAT32)
        .value("FLOAT64", DType::FLOAT64)
        .export_values();

    nb::class_<Array>(m, "Array")
        .def_prop_ro("device", &Array::device)
        .def_ro("dtype", &Array::dtype)
        .def_ro("shape", &Array::shape)
        .def("__repr__", &Array::repr);

    m.def("from_numpy", &from_numpy<float>, nb::arg("owner"));
    m.def("from_numpy", &from_numpy<double>, nb::arg("owner"));
    m.def("from_numpy", &from_numpy<std::int32_t>, nb::arg("owner"));
    m.def("from_numpy", &from_numpy<std::int64_t>, nb::arg("owner"));
    m.def("from_numpy", &from_numpy<bool>, nb::arg("owner"));
}
