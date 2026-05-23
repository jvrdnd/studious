#pragma once

#include <cstddef>
#include <nanobind/nanobind.h>
#include <stdexcept>

enum class Device { CPU, CUDA, METAL };

enum class DType { BOOL, INT32, INT64, BFLOAT16, FLOAT32, FLOAT64 };
inline constexpr std::size_t dtype_size(DType dtype) {
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

    throw std::invalid_argument("unknown dtype");
}

class Buffer {
public:
    virtual ~Buffer() noexcept = default;

    const Device device;
    void *const ptr;
    const std::size_t nbytes;

    Buffer(const Buffer &) = delete; // copy construction: Buffer b = a
    Buffer &operator=(const Buffer &) = delete; // copy assignment: b = a
    Buffer(Buffer &&) = delete; // move construction: Buffer b = std::move(a)
    Buffer &operator=(Buffer &&) = delete; // move assignment: b = std::move(a)

protected:
    Buffer(Device device, void *ptr, std::size_t nbytes) noexcept;
};

class CpuBuffer final : public Buffer {
public:
    explicit CpuBuffer(std::size_t nbytes);
    ~CpuBuffer() noexcept override;

private:
    static void *allocate(std::size_t nbytes);
};

class NumPyBuffer final : public Buffer {
public:
    NumPyBuffer(void *ptr, std::size_t nbytes, nanobind::object numpy_ptr);
    ~NumPyBuffer() noexcept override;

private:
    nanobind::object numpy_ptr; // ensure NumPy does not free the memory
};
