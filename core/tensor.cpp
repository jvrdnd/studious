#include "tensor.h"

#include <cassert>
#include <sstream>

Tensor::Tensor(const std::vector<double> &data_, const std::vector<std::size_t> &shape_) : shape(shape_), data(data_) {}

std::string Tensor::repr() const {
  std::ostringstream os;

  std::function<void(std::size_t, std::size_t)> repr_at{ [&](std::size_t dim, std::size_t offset) {
    os << "[";

    if (dim + 1 == shape.size()) {
      // leaf: print scalars
      for (std::size_t i{ 0 }; i < shape[dim]; ++i) {
        os << data[offset + i];
        if (i + 1 < shape[dim]) {
          os << ", ";
        }
      }

    } else {
      // recursion: repr subarrays
      std::size_t stride{ 1 };
      for (std::size_t i{ dim + 1 }; i < shape.size(); ++i) {
        stride *= shape[i];
      }
      for (std::size_t i{ 0 }; i < shape[dim]; ++i) {
        repr_at(dim + 1, offset + i * stride);
        if (i + 1 < shape[dim]) {
          os << ", ";
        }
      }
    }

    os << "]";
  } };

  os << "array(";
  if (shape.empty()) {
    os << data[0];
  } else {
    repr_at(0, 0);
  }
  os << ")";

  return os.str();
}
