#include <cstddef>
#include <new>

#include "device.hpp"

namespace sx::Cpu {

constexpr std::size_t Alignment = 64;
static_assert((Alignment & (Alignment - 1)) == 0);

std::byte *Device::allocate(std::size_t size) const {
    if (size == 0) {
        return nullptr;
    }
    return static_cast<std::byte *>(::operator new(size, std::align_val_t{Alignment}));
}

void Device::deallocate(std::byte *ptr) const noexcept {
    ::operator delete(ptr, std::align_val_t{Alignment});
}

} // namespace sx::Cpu
