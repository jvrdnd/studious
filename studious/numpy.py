from ._lib import CpuDevice, DType, MetalDevice, make_array
from .core.Array import Array


def array(data: object, dtype: DType | None = None, *, device: CpuDevice | MetalDevice | None = None) -> Array:
    binding = make_array(data, dtype=dtype, device=device)
    return Array(binding)
