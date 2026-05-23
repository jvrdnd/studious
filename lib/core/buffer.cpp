#include <cstdlib>
#include <nanobind/nanobind.h>
#include <new>
#include <utility>

#include "buffer.hpp"

Buffer::Buffer(Device device, void *ptr, std::size_t nbytes) noexcept :
    device{ device }, ptr{ ptr }, nbytes{ nbytes } {}

CpuBuffer::CpuBuffer(std::size_t nbytes) : Buffer{ Device::CPU, allocate(nbytes), nbytes } {}

CpuBuffer::~CpuBuffer() noexcept {
    std::free(ptr);
}

void *CpuBuffer::allocate(std::size_t nbytes) {
    if (nbytes == 0) {
        return nullptr;
    }

    void *p = std::malloc(nbytes);
    if (!p) {
        throw std::bad_alloc();
    }

    return p;
}

NumPyBuffer::NumPyBuffer(void *ptr, std::size_t nbytes, nanobind::object numpy_ptr) :
    Buffer{ Device::CPU, ptr, nbytes }, numpy_ptr{ std::move(numpy_ptr) } {}

NumPyBuffer::~NumPyBuffer() noexcept {
    nanobind::gil_scoped_acquire gil; // acquire Python GIL to safely update the reference count
    numpy_ptr.reset();
}
