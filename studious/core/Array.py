from __future__ import annotations

from dataclasses import dataclass
from types import CapsuleType

from .._lib import CpuArray, MetalArray


def unpack(self: Array):
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError("numpy required for unpack methods") from None
    return np.from_dlpack(self)


@dataclass(frozen=True)
class Array:
    _binding: CpuArray | MetalArray

    def __getattr__(self, name: str):
        return getattr(self._binding, name)

    def __dlpack__(self, *, stream: object | None = None) -> CapsuleType:
        return self._binding.__dlpack__(stream=stream)

    def __dlpack_device__(self) -> tuple[int, int]:
        return self._binding.__dlpack_device__()

    def __repr__(self) -> str:
        return repr(unpack(self))

    def __str__(self) -> str:
        return str(unpack(self))
