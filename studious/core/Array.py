from __future__ import annotations

from dataclasses import dataclass
from types import CapsuleType

from .._lib import (
    CpuArray,
    Device,
    DType,
    MetalArray,
    dlpack_get_data,
    dlpack_get_device,
    dlpack_get_dtype,
    dlpack_get_shape,
    dlpack_get_strides,
)


@dataclass(frozen=True)
class Array:
    _binding: CpuArray | MetalArray

    def __dlpack__(self, *, stream: object | None = None) -> CapsuleType:
        return self._binding.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._binding.__dlpack_device__()

    def __repr__(self) -> str:
        return repr(self.data)

    def __str__(self) -> str:
        return str(self.data)

    @property
    def device(self) -> Device:
        return dlpack_get_device(self)

    @property
    def dtype(self) -> DType:
        return dlpack_get_dtype(self)

    @property
    def shape(self) -> list[int]:
        return dlpack_get_shape(self)

    @property
    def strides(self) -> list[int]:
        return dlpack_get_strides(self)

    @property
    def data(self) -> list[bool] | list[int] | list[float]:
        return dlpack_get_data(self)
