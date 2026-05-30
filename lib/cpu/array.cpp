#include "array.hpp"
#include "buffer.hpp"
#include "dtype.hpp"

namespace sx::Cpu {

[[nodiscard]] std::shared_ptr<Array<Device>>
make_array(nb::handle data, std::optional<DType> dtype, std::shared_ptr<const Device> device) {
    ArraySpec traits{infer_array_spec(data, dtype)};

    std::shared_ptr<const Buffer> buffer = std::make_shared<Buffer>(
        std::dynamic_pointer_cast<const Device>(device),
        dtype_size(traits.dtype) * traits.size
    );
    memcpy(buffer->data(), data, dispatch_dtype, traits.dtype);

    return std::make_shared<Array<Device>>(traits.dtype, traits.size, std::move(traits.shape), buffer);
}

} // namespace sx::Cpu
