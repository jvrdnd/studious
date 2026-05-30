#include "buffer.hpp"

namespace sx {

namespace nb = nanobind;

std::size_t memcpy(std::byte *ptr, nb::handle data, DType dtype) {
    std::size_t offset;

    if (nb::isinstance<nb::list>(data)) {
        offset = 0;
        for (nb::handle item : nb::borrow<nb::list>(data)) {
            offset += memcpy(ptr + offset, item, dtype);
        }

    } else {
        dispatch_dtype(dtype, [&]<typename T>() {
            T scalar = nb::cast<T>(data);
            offset = sizeof(T);
            std::memcpy(ptr, &scalar, offset);
        });
    }

    return offset;
}

} // namespace sx
