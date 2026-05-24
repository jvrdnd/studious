#include <cstdlib>

#include "buffer.hpp"

Cpu::Buffer::Buffer(std::shared_ptr<const Cpu::Device> device, std::size_t size) :
    ::Buffer{ device }, data_{ device->allocate(size) }, size_{ size } {}

Cpu::Buffer::~Buffer() noexcept {
    std::static_pointer_cast<const Cpu::Device>(device)->deallocate(data_);
}

std::byte *Cpu::Buffer::data() const noexcept {
    return data_;
}

std::size_t Cpu::Buffer::size() const noexcept {
    return size_;
}
