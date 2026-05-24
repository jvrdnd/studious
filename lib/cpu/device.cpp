#include <cstdlib>
#include <new>
#include <string>

#include "device.hpp"

Platform Cpu::Device::platform() const noexcept {
    return Platform::CPU;
}

std::byte *Cpu::Device::allocate(std::size_t nbytes) const {
    if (nbytes == 0) {
        return nullptr;
    }

    std::byte *ptr = static_cast<std::byte *>(std::malloc(nbytes));
    if (ptr == nullptr) {
        throw std::bad_alloc{};
    }

    return ptr;
}

void Cpu::Device::deallocate(std::byte *ptr) const noexcept {
    std::free(ptr);
}

std::string Cpu::Device::repr_() const {
    return "";
}
