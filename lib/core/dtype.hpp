#include <cstddef>
#include <string_view>

enum class DType { Bool, Int32, Int64, UInt8, Float16, Float32, BFloat16 };

constexpr std::size_t dtype_size(DType dtype);

constexpr std::string_view dtype_name(DType dtype);
