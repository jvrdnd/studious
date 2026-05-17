#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

class Tensor;
using TensorPtr = std::shared_ptr<Tensor>;

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
  std::vector<std::size_t> shape;

  Tensor(const std::vector<double> &data_, const std::vector<std::size_t> &shape_);

  std::string repr() const;

private:
  std::vector<double> data;
};