#include <cstddef>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>
#include <span>
#include <vector>

enum class Device { CPU, CUDA, METAL };

enum class DType { BOOL, INT32, INT64, BFLOAT16, FLOAT32, FLOAT64 };
constexpr std::size_t dtype_size(DType dtype) {
    switch (dtype) {
        case DType::BOOL:
            return 1;
        case DType::BFLOAT16:
            return 2;
        case DType::INT32:
        case DType::FLOAT32:
            return 4;
        case DType::INT64:
        case DType::FLOAT64:
            return 8;
    }
}

std::vector<std::int64_t> default_strides(std::span<const std::int64_t> shape, DType dtype) {
    std::vector<std::int64_t> strides(shape.size());
    std::int64_t stride{ static_cast<std::int64_t>(dtype_size(dtype)) };

    for (std::size_t i = shape.size(); i-- > 0;) {
        strides[i] = stride;
        stride *= shape[i];
    }

    return strides;
}

struct Buffer {
    const Device device;
    void *const ptr;
    const std::size_t nbytes = 0;
    void (*const deleter)(void *ptr) noexcept;

    Buffer(Device device, void *ptr, std::size_t nbytes, void (*deleter)(void *ptr) noexcept) :
        device(device), ptr(ptr), nbytes(nbytes), deleter(deleter) {
        assert(ptr != nullptr || nbytes == 0);
        assert(deleter != nullptr);
    }

    ~Buffer() {
        if (ptr) {
            deleter(ptr);
        }
    }

    // prevent double-free
    Buffer(const Buffer &) = delete; // no Buffer b = a
    Buffer &operator=(const Buffer &) = delete; // no b = a

    // prevent unsafe ownership transfer
    Buffer(Buffer &&) = delete; // no Buffer b = std::move(a)
    Buffer &operator=(Buffer &&) = delete; // no b = std::move(a)
};

struct Array {
    DType dtype;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    std::int64_t offset_bytes = 0;
    std::shared_ptr<Buffer> buffer;

    Array(DType dtype, std::vector<std::int64_t> shape, std::shared_ptr<Buffer> buffer) :
        dtype(dtype), shape(std::move(shape)), strides(default_strides(this->shape, dtype)), buffer(std::move(buffer)) {
    }

    Array(DType dtype,
        std::vector<std::int64_t> shape,
        std::vector<std::int64_t> strides,
        std::shared_ptr<Buffer> buffer,
        std::int64_t offset_bytes = 0) :
        dtype(dtype), shape(std::move(shape)), strides(std::move(strides)), offset_bytes(offset_bytes),
        buffer(std::move(buffer)) {}

    Device device() const {
        return buffer->device;
    }
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

    nanobind::class_<Array>(m, "Array")
        .def_prop_ro("device", &Array::device)
        .def_ro("dtype", &Array::dtype)
        .def_ro("shape", &Array::shape);
}