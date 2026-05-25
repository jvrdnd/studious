#include <cstddef>
#include <new>
#include <string>

#include "device.hpp"

Platform Cpu::Device::platform() const noexcept {
    return Platform::Cpu;
}

namespace {
constexpr std::size_t Alignment = 64;
static_assert((Alignment & (Alignment - 1)) == 0);
} // namespace

std::byte *Cpu::Device::allocate(std::size_t size) const {
    if (size == 0) {
        return nullptr;
    }
    return static_cast<std::byte *>(::operator new(size, std::align_val_t{ Alignment }));
}

void Cpu::Device::deallocate(std::byte *ptr) const noexcept {
    ::operator delete(ptr, std::align_val_t{ Alignment });
}

std::string Cpu::Device::repr_() const {
    return "";
}
