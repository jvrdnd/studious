#include <Metal/MTLLibrary.hpp>
#include <dlfcn.h>
#include <filesystem>
#include <new>
#include <stdexcept>

#include "device.hpp"
#include "kernels/kernels.hpp"

namespace sx::Metal {

NS::SharedPtr<MTL::Library> Device::metallib() {
    Dl_info info{};
    if (!dladdr(reinterpret_cast<const void *>(&dispatch_kernel), &info)) {
        throw std::runtime_error{"unable to locate loaded extension path"};
    }
    const std::filesystem::path fs_path{std::filesystem::path{info.dli_fname}.parent_path() / METALLIB_FILENAME};
    NS::SharedPtr<NS::String> ns_path =
        NS::TransferPtr(NS::String::alloc()->init(fs_path.c_str(), NS::UTF8StringEncoding));

    NS::Error *error = nullptr;
    NS::SharedPtr<MTL::Library> library{NS::TransferPtr(handle_->newLibrary(ns_path.get(), &error))};
    if (!library) {
        std::string msg = "unable to load metal library: ";
        msg.append(fs_path);
        if (error != nullptr) {
            msg += ": ";
            msg += error->localizedDescription()->utf8String();
        }
        throw std::runtime_error{msg};
    }

    return library;
}

Device::Device(std::int32_t id, MTL::Device *handle) : sx::Device{id} {
    if (handle == nullptr) {
        throw std::invalid_argument{"invalid device"};
    }
    handle_ = NS::TransferPtr(handle->retain());

    MTL::CommandQueue *queue = handle_->newCommandQueue();
    if (queue == nullptr) {
        throw std::runtime_error{"unable to create command queue"};
    }
    queue_ = NS::TransferPtr(queue);

    library_ = metallib();
}

MTL::Buffer *Device::allocate(std::size_t nbytes, StorageMode mode) const {
    if (nbytes == 0) {
        return nullptr;
    }

    MTL::Buffer *buffer{handle_->newBuffer(
        nbytes,
        mode == StorageMode::Private ? MTL::ResourceStorageModePrivate : MTL::ResourceStorageModeShared
    )};
    if (buffer == nullptr) {
        throw std::bad_alloc{};
    }

    return buffer;
}

void Device::deallocate(MTL::Buffer *buffer) const noexcept {
    if (buffer != nullptr) {
        buffer->release();
    }
}

} // namespace sx::Metal
