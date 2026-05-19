#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include "tensor.h"

namespace nb = nanobind;

static std::vector<std::size_t> infer_shape(nb::handle obj) {
  if (!nb::isinstance<nb::list>(obj)) {
    nb::cast<double>(obj); // validate scalar
    return {};
  }

  nb::list list = nb::cast<nb::list>(obj);
  if (nb::len(list) == 0) {
    return { 0 };
  }

  std::vector<std::size_t> child_shape = infer_shape(list[0]);

  for (std::size_t i = 1; i < nb::len(list); ++i) {
    std::vector<std::size_t> child_sibling_shape = infer_shape(list[i]);

    if (child_sibling_shape != child_shape) {
      throw std::runtime_error("requested array has an inhomogeneous shape");
    }
  }

  std::vector<std::size_t> shape;
  shape.push_back(static_cast<std::size_t>(nb::len(list)));
  shape.insert(shape.end(), child_shape.begin(), child_shape.end());

  return shape;
}

static void flatten(nb::handle obj, std::vector<double> &out) {
  if (!nb::isinstance<nb::list>(obj)) {
    out.push_back(nb::cast<double>(obj));
    return;
  }

  nb::list list = nb::cast<nb::list>(obj);

  for (nb::handle item : list) {
    flatten(item, out);
  }
}

static Tensor create_tensor(nb::handle obj) {
  std::vector<std::size_t> shape = infer_shape(obj);

  std::vector<double> data;
  flatten(obj, data);

  return Tensor(data, shape);
}

NB_MODULE(_core, m) {
  nb::class_<Tensor>(m, "Tensor").def_rw("shape", &Tensor::shape).def("__repr__", &Tensor::repr);
  m.def("create_tensor", &create_tensor);
}