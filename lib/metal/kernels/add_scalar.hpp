#include <Metal/Metal.hpp>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

#include "../../core/array.hpp"
#include "../../core/dtype.hpp"
#include "../buffer.hpp"
#include "../device.hpp"

namespace sx::Metal {

namespace {
inline NS::SharedPtr<NS::String> instance_name(std::string_view template_name, Dtype dtype) {
    std::string instance_name{std::string(template_name) + "__" + std::string(dtype_name(dtype))};
    return NS::TransferPtr(NS::String::alloc()->init(instance_name.c_str(), NS::UTF8StringEncoding));
}
} // namespace

template <typename T> inline Array<Buffer> add_scalar(Array<Buffer> &in, T scalar) {
    Dtype dtype{in.dtype()};
    NS::SharedPtr<NS::String> name{instance_name("add_scalar", dtype)};

    std::shared_ptr<const sx::Metal::Device> device{in.buffer()->device()};
    NS::SharedPtr<MTL::Function> f{NS::TransferPtr(device->library()->newFunction(name.get()))};
    if (!f) {
        throw std::runtime_error(std::string("metal function not found: ") + name->utf8String());
    }

    NS::Error *error = nullptr;
    NS::SharedPtr<MTL::ComputePipelineState> pipeline{
        NS::TransferPtr(device->handle()->newComputePipelineState(f.get(), &error))
    };

    if (!pipeline) {
        std::string msg = "could not create compute pipeline";
        if (error != nullptr) {
            msg += ": ";
            msg += error->localizedDescription()->utf8String();
        }
        throw std::runtime_error{msg};
    }

    MTL::CommandBuffer *command_buffer = device->queue()->commandBuffer();
    MTL::ComputeCommandEncoder *command_encoder = command_buffer->computeCommandEncoder();

    command_encoder->setComputePipelineState(pipeline.get());

    const std::size_t nbytes{in.buffer()->size()};
    const auto size{static_cast<std::uint32_t>(nbytes / dtype_size(dtype))}; // must map to uint in the metal kernel

    Array<Buffer> out{dtype, in.shape(), in.strides(), std::make_shared<Buffer>(device, nbytes)};
    command_encoder->setBuffer(in.buffer()->handle(), 0, 0);
    command_encoder->setBuffer(out.buffer()->handle(), 0, 1);
    command_encoder->setBytes(&scalar, sizeof(scalar), 2);
    command_encoder->setBytes(&size, sizeof(size), 3);

    NS::UInteger width{pipeline->threadExecutionWidth()};
    if (width == 0) {
        width = 1;
    }

    MTL::Size threads_per_threadgroup = MTL::Size::Make(width, 1, 1);
    MTL::Size threads_per_grid = MTL::Size::Make(size, 1, 1);

    command_encoder->dispatchThreads(threads_per_grid, threads_per_threadgroup);
    command_encoder->endEncoding();

    command_buffer->commit();
    command_buffer->waitUntilCompleted();

    return out;
}

} // namespace sx::Metal
