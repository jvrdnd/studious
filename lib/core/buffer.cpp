#include "buffer.hpp"

Buffer::Buffer(std::shared_ptr<const Device> device) : device{ std::move(device) } {}
