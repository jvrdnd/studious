#include "buffer.hpp"

Metal::Buffer::Buffer(std::shared_ptr<const Metal::Device> device, std::size_t size) :
    ::Buffer{ device }, handle_{ device->allocate(size) } {}

Metal::Buffer::~Buffer() noexcept {
    std::static_pointer_cast<const Metal::Device>(device)->deallocate(handle_);
}

std::byte *Metal::Buffer::data() const noexcept {
    return static_cast<std::byte *>(handle_->contents());
}

std::size_t Metal::Buffer::size() const noexcept {
    return handle_->length();
}
