from __future__ import annotations

from dataclasses import dataclass

from .array import Array
from .interpreter import EvalInterpreter, interpreter_context
from .jit import execute_rule
from .types import AbstractValue, Input, Instruction, Param, PrimitiveLike


@dataclass
class Primitive(PrimitiveLike):
    name: str

    def __hash__(self) -> int:
        return hash(self.name)

    def evaluate(self, *inputs: Array, **params: Param) -> tuple[Array, ...]:
        return execute_rule(self, **params)(*inputs)

    def interpret(self, *inputs: AbstractValue, **params: Param) -> tuple[AbstractValue, ...]:
        raise NotImplementedError(f"no interpreter implementation for '{self.name}'")

    def bind(self, *inputs: Input, **params: Param) -> tuple[Input, ...]:
        interpreter = interpreter_context().interpreter

        return interpreter.process(
            Instruction(
                self,
                inputs
                if isinstance(interpreter, EvalInterpreter)
                else tuple(interpreter.to_level(input) for input in inputs),
                params,
            )
        )
