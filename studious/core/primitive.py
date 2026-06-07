from __future__ import annotations

from dataclasses import dataclass

from .array import Array
from .base import Input, Instruction, Param, Placeholder, PrimitiveLike, Tracer
from .jit import execute_rule
from .trace import trace_context


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
        # catch leaked tracers
        for input in inputs:
            if isinstance(input, Tracer) and not input.trace.is_valid:
                raise RuntimeError(f"detected escaped tracer: {input}")

        return trace_context().trace.process(Instruction(self, inputs, params))
