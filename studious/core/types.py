from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .._lib import Dtype
from .array import Array


@dataclass(frozen=True)
class Placeholder:
    dtype: Dtype
    shape: tuple[int, ...]


class PrimitiveLike(Protocol):
    name: str

    def execute(self, *inputs: Array, **params: Param) -> tuple[Array, ...]: ...
    def trace(self, *inputs: Placeholder, **params: Param) -> tuple[Placeholder, ...]: ...
    def bind(self, *inputs: Input, **params: Param) -> tuple[Input, ...]: ...


class TraceLike(Protocol):
    level: int
    is_valid: bool


@runtime_checkable
class Tracer(Protocol):
    @property
    def trace(self) -> TraceLike: ...

    @property
    def placeholder(self) -> Placeholder: ...


type Input = Array | Tracer
type Param = bool | int | float | str | Dtype | tuple[Param, ...] | None


class Traceable(Protocol):
    def __call__(self, *inputs: Tracer, **params: Param) -> tuple[Tracer, ...]: ...


class Executable(Protocol):
    def __call__(self, *inputs: Array) -> tuple[Array, ...]: ...


class ExecuteRule(Protocol):
    def __call__(self, primitive: PrimitiveLike, **params: Param) -> Executable: ...


@dataclass(frozen=True)
class Instruction:
    primitive: PrimitiveLike
    inputs: tuple[Input, ...]
    params: dict[str, Param]


@dataclass(frozen=True)
class ArrayInstruction(Instruction):
    inputs: tuple[Array, ...]
