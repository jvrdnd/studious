#pragma once

#include <Metal/MTLComputePipeline.hpp>
#include <Metal/Metal.hpp>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

#include "../core/device.hpp"

namespace sx::Metal {

enum class StorageMode { Shared, Private };

class Device final : public sx::Device {
public:
    explicit Device(std::int32_t id, MTL::Device *handle);
    ~Device() noexcept override = default;

    [[nodiscard]] Platform platform() const noexcept override {
        return Platform::Metal;
    }

    [[nodiscard]] MTL::Buffer *allocate(std::size_t nbytes, StorageMode mode) const;
    void deallocate(MTL::Buffer *buffer) const noexcept;

    [[nodiscard]] MTL::Device *handle() const noexcept {
        return handle_.get();
    }
    [[nodiscard]] MTL::CommandQueue *queue() const noexcept {
        return queue_.get();
    }

    [[nodiscard]] MTL::Library *library() const noexcept {
        return library_.get();
    }

    [[nodiscard]] MTL::ComputePipelineState *kernel(std::string name) const;

private:
    NS::SharedPtr<MTL::Device> handle_;
    NS::SharedPtr<MTL::CommandQueue> queue_;

    NS::SharedPtr<MTL::Library> library_;
    [[nodiscard]] NS::SharedPtr<MTL::Library> metallib();

    mutable std::unordered_map<std::string, NS::SharedPtr<MTL::ComputePipelineState>> kernels_;
    mutable std::mutex kernels_mutex_;
};

} // namespace sx::Metal
