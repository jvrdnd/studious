#include "buffer.hpp"

Buffer::Buffer(Device device, void *ptr) noexcept : device{ device }, ptr{ ptr } {}

CpuBuffer::CpuBuffer(std::size_t nbytes) : Buffer{ Device::CPU, allocate(nbytes) }, nbytes{ nbytes } {}

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

NumPyBuffer::NumPyBuffer(void *ptr, nanobind::object owner) : Buffer{ Device::CPU, ptr }, owner{ std::move(owner) } {}

NumPyBuffer::~NumPyBuffer() noexcept {
    nanobind::gil_scoped_acquire gil; // acquire Python GIL to safely update the reference count
    owner.reset();
}
