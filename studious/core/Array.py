from __future__ import annotations

from dataclasses import dataclass
from types import CapsuleType

from .._lib import CpuArray, DType, MetalArray, from_dlpack


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
    def dtype(self) -> DType:
        return from_dlpack(self).dtype

    @property
    def shape(self) -> list[int]:
        return from_dlpack(self).shape

    @property
    def strides(self) -> list[int]:
        return from_dlpack(self).strides

    @property
    def data(self) -> list[bool] | list[int] | list[float]:
        return from_dlpack(self).data
