from __future__ import annotations

from dataclasses import dataclass
from types import CapsuleType

from .._lib import (
    CpuArray,
    Device,
    Dtype,
    MetalArray,
    dlpack_get_data,
    dlpack_get_device,
    dlpack_get_dtype,
    dlpack_get_shape,
    dlpack_get_strides,
)

type Shaped = bool | int | float | list[Shaped]


def reshape(array: Array) -> Shaped:
    def build(offset: int, dims: tuple[int, ...]) -> tuple[Shaped, int]:
        if not dims:
            return array.data[offset], offset + 1
        size = dims[0]
        result: Shaped = []
        for _ in range(size):
            item, offset = build(offset, dims[1:])
            result.append(item)
        return result, offset

    out, _ = build(0, array.shape)
    return out


def format(x: Shaped, indent: int = 0) -> str:
    if not isinstance(x, list) or not isinstance(x[0], list):
        return repr(x)

    inner_indent = indent + 2
    pieces = [" " * inner_indent + format(item, inner_indent) for item in x]

    return "[\n" + ",\n".join(pieces) + "\n" + " " * indent + "]"


@dataclass(frozen=True)
class Array:
    _binding: CpuArray | MetalArray

    def __dlpack__(self, *, stream: object | None = None) -> CapsuleType:
        return self._binding.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._binding.__dlpack_device__()

    def __repr__(self) -> str:
        return format(reshape(self))

    def __str__(self) -> str:
        return self.__repr__()

    @property
    def device(self) -> Device:
        return dlpack_get_device(self)

    @property
    def dtype(self) -> Dtype:
        return dlpack_get_dtype(self)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(dlpack_get_shape(self))

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(dlpack_get_strides(self))

    @property
    def data(self) -> list[bool] | list[int] | list[float]:
        return dlpack_get_data(self)
