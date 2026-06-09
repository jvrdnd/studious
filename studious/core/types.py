from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .._lib import Dtype
from .array import Array


class InterpreterLike(Protocol):
    level: int
    is_valid: bool


@dataclass(frozen=True)
class AbstractValue:
    dtype: Dtype
    shape: tuple[int, ...]


@runtime_checkable
class Placeholder(Protocol):
    @property
    def interpreter(self) -> InterpreterLike: ...

    @property
    def abstract_value(self) -> AbstractValue: ...


type Input = Array | Placeholder
type Param = bool | int | float | str | Dtype | tuple[Param, ...] | None


class Evaluatable(Protocol):
    def __call__(self, *inputs: Array) -> tuple[Array, ...]: ...


class Interpretable(Protocol):
    def __call__(self, *inputs: Placeholder, **params: Param) -> tuple[Placeholder, ...]: ...


class PrimitiveLike(Protocol):
    name: str

    def evaluate(self, *inputs: Array, **params: Param) -> tuple[Array, ...]: ...
    def interpret(self, *inputs: AbstractValue, **params: Param) -> tuple[AbstractValue, ...]: ...
    def bind(self, *inputs: Input, **params: Param) -> tuple[Input, ...]: ...


class EvalRule(Protocol):
    def __call__(self, primitive: PrimitiveLike, **params: Param) -> Evaluatable: ...


@dataclass(frozen=True)
class Instruction:
    primitive: PrimitiveLike
    inputs: tuple[Input, ...]
    params: dict[str, Param]


@dataclass(frozen=True)
class EvalInstruction(Instruction):
    inputs: tuple[Array, ...]
