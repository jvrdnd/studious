from __future__ import annotations

from dataclasses import dataclass

from .array import Array
from .jit import execute_rule
from .trace import EvalTrace, trace_context
from .types import Input, Instruction, Param, Placeholder, PrimitiveLike


@dataclass
class Primitive(PrimitiveLike):
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def execute(self, *inputs: Array, **params: Param) -> tuple[Array, ...]:
        return execute_rule(self, **params)(*inputs)

    def trace(self, *inputs: Placeholder, **params: Param) -> tuple[Placeholder, ...]:
        raise NotImplementedError(f"no trace implementation for '{self.name}'")

    def bind(self, *inputs: Input, **params: Param) -> tuple[Input, ...]:
        trace = trace_context().trace

        return trace.process(
            Instruction(
                self,
                inputs if isinstance(trace, EvalTrace) else tuple(trace.to_level(input) for input in inputs),
                params,
            )
        )
