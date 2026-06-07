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
inline std::string instance_name(std::string_view template_name, Dtype dtype) {
    return std::string(template_name) + "__" + std::string(dtype_name(dtype));
}
} // namespace

template <typename T> inline Array<Buffer> add_scalar(Array<Buffer> &in, T scalar) {
    const Dtype dtype{in.dtype()};

    const std::shared_ptr<const sx::Metal::Device> device{in.buffer()->device()};
    const NS::SharedPtr<MTL::ComputePipelineState> pipeline{
        NS::TransferPtr(device->kernel(instance_name("add_scalar", dtype)))
    };

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
