from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .array import Array
from .input import Input, Param, Placeholder, Tracer
from .jit import Executable, jit
from .trace import trace_context
from .util import cache


@dataclass
class Primitive:
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def execute(self, *inputs: Array, **params: Param) -> tuple[Array, ...]:
        return execute_rule(self, **params)(*inputs)

    def trace(self, *inputs: Placeholder, **params: Param) -> tuple[Placeholder, ...]:
        raise NotImplementedError(f"no trace implementation for '{self.name}'")

    def bind(self, *inputs: Input, **params: Param) -> tuple[Input, ...]:
        # catch leaked tracers
        for input in inputs:
            if isinstance(input, Tracer) and not input.trace.is_valid:
                raise RuntimeError(f"detected escaped tracer: {input}")

        return trace_context().trace.process(self, inputs, params)


class ExecuteRule(Protocol):
    def __call__(self, primitive: Primitive, **params: Param) -> Executable: ...


@cache()
def execute_rule(primitive: Primitive, **params: Param) -> Executable:
    def function(*inputs: Input) -> tuple[Input, ...]:
        return primitive.bind(*inputs, **params)

    return jit(function)  # type: ignore : input is tracer + params in closure
