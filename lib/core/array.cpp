#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

enum class Device { CPU, CUDA, METAL };
enum class DType { BOOL, INT32, INT64, BFLOAT16, FLOAT32, FLOAT64 };

struct Array {
    const Device device;
    const DType dtype;
    const std::vector<int64_t> shape;

    const void *data;

    Array(Device device, DType dtype, std::vector<int64_t> shape, void *data) :
        device(device), dtype(dtype), shape(std::move(shape)), data(data) {}
};

NB_MODULE(_lib, m) {
    nanobind::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("CUDA", Device::CUDA)
        .value("METAL", Device::METAL)
        .export_values();

    nanobind::enum_<DType>(m, "DType")
        .value("BOOL", DType::BOOL)
        .value("INT32", DType::INT32)
        .value("INT64", DType::INT64)
        .value("BFLOAT16", DType::BFLOAT16)
        .value("FLOAT32", DType::FLOAT32)
        .value("FLOAT64", DType::FLOAT64)
        .export_values();

    nanobind::class_<Array>(m, "Array").def_ro("device", &Array::device).def_ro("shape", &Array::shape);
}