#include "../cpu/dtype.hpp"
#include "bindings.hpp"
#include "platform.hpp"

std::shared_ptr<::BaseArray>
make_array(std::shared_ptr<const Device> device, nanobind::iterable iterable, DType dtype) {
    switch (device->platform()) {
        case Platform::Cpu:
            return Cpu::dispatch_dtype(dtype, [&]<typename T>() -> std::shared_ptr<::BaseArray> {
                return std::make_shared<Cpu::Array<T>>(
                    std::move(std::static_pointer_cast<const Cpu::Device>(device)), to_vector<T>(iterable)
                );
            });
        case Platform::Metal:
            break;
    }
    throw std::invalid_argument{ "invalid device" };
}
