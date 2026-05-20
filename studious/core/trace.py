from __future__ import annotations

from dataclasses import dataclass

from .._lib import Array, DType
from .primitive import Primitive


@dataclass
class Trace:
    level: int
    is_valid: bool = True

    def process_primitive(
        self, primitive: Primitive, args: tuple[Traceable, ...], params: dict[str, object]
    ) -> Traceable:
        raise NotImplementedError


@dataclass
class Aval:
    dtype: DType
    shape: tuple[int, ...]


@dataclass
class Tracer:
    aval: Aval
    trace: Trace


Scalar = bool | float | int
Traceable = Scalar | Array | Tracer


class EagerTrace(Trace):
    def __init__(self):
        super().__init__(level=0)

    def process_primitive(self, primitive: Primitive, args: tuple[Traceable, ...], params: dict[str, object]) -> Array:
        return primitive.impl(*args, **params)  # type: ignore[arg-type]: called with tuple[Array, ...] args


@dataclass
class TraceContext:
    stack: list[Trace] = [EagerTrace()]

    @property
    def trace(self) -> Trace:
        return self.stack[-1]
