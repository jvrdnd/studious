from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from functools import lru_cache, wraps

from .array import Array
from .interpreter import Interpreter, interpreter_context, interpreter_scope
from .types import (
    AbstractValue,
    EvalRule,
    Evaluatable,
    Input,
    Instruction,
    Interpretable,
    Param,
    Placeholder,
    PrimitiveLike,
)


@dataclass
class JitInterpreter(Interpreter):
    instructions: list[JitPlaceholderInstruction] = field(default_factory=lambda: [])

    def array_to_level(self, array: Array) -> JitPlaceholder:
        return JitPlaceholder(self, AbstractValue(array.dtype, array.shape))

    def placeholder_to_level(self, placeholder: Placeholder) -> JitPlaceholder:
        return JitPlaceholder(self, placeholder.abstract_value)

    def process(self, instruction: JitPlaceholderInstruction) -> tuple[Placeholder, ...]:  # type: ignore
        self.instructions.append(instruction)
        in_placeholders = [input.abstract_value for input in instruction.inputs]
        out_placeholders = instruction.primitive.interpret(*in_placeholders, **instruction.params)
        return tuple(JitPlaceholder(self, placeholder) for placeholder in out_placeholders)


@dataclass
class JitPlaceholder(Placeholder):
    _interpreter: JitInterpreter
    _placeholder: AbstractValue

    @property
    def interpreter(self) -> JitInterpreter:
        return self._interpreter

    @property
    def abstract_value(self) -> AbstractValue:
        return self._placeholder


@dataclass(frozen=True)
class JitPlaceholderInstruction(Instruction):
    inputs: tuple[JitPlaceholder, ...]


def compile(instructions: list[JitPlaceholderInstruction], placeholders: tuple[Placeholder, ...]) -> Evaluatable:
    raise NotImplementedError


def jit(function: Interpretable) -> Evaluatable:
    @wraps(function)
    def wrapper(*inputs: Array, **params: Param) -> tuple[Array, ...]:
        # push the jit interpreter
        with interpreter_scope(JitInterpreter) as interpreter:
            # gather ops
            in_placedholders = tuple(
                JitPlaceholder(interpreter, AbstractValue(input.dtype, input.shape)) for input in inputs
            )
            out_placeholders = function(*in_placedholders, **params)

        return compile(interpreter.instructions, out_placeholders)(*inputs)

    return wrapper


def cache(max_size: int = 4096) -> Callable[[EvalRule], EvalRule]:
    def wrap(function: EvalRule) -> EvalRule:
        @lru_cache(maxsize=max_size)
        def cached(_: Hashable, primitive: PrimitiveLike, **params: Param) -> Evaluatable:
            return function(primitive, **params)

        @wraps(function)
        def wrapper(primitive: PrimitiveLike, **params: Param) -> Evaluatable:
            return cached(interpreter_context().key, primitive, **params)

        return wrapper

    return wrap


@cache()
def execute_rule(primitive: PrimitiveLike, **params: Param) -> Evaluatable:
    def function(*inputs: Input) -> tuple[Input, ...]:
        return primitive.bind(*inputs, **params)

    return jit(function)  # type: ignore : input is placeholder + params in closure
