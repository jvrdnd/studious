from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from .array import Array
from .types import EvalInstruction, Input, Instruction, InterpreterLike, Placeholder


@dataclass
class Interpreter(InterpreterLike, ABC):
    level: int
    is_valid: bool = True

    def array_to_level(self, array: Array) -> Placeholder:
        raise NotImplementedError

    def placeholder_to_level(self, placeholder: Placeholder) -> Placeholder:
        raise NotImplementedError

    def to_level(self, input: Input) -> Placeholder:
        if isinstance(input, Array):
            return self.array_to_level(input)

        if not input.interpreter.is_valid:
            # catch leaked placeholders
            raise RuntimeError(f"detected escaped placeholder: {input}")

        if input.interpreter is self:
            return input

        if input.interpreter.level >= self.level:
            raise RuntimeError("placeholder must be from a lower level")

        return self.placeholder_to_level(input)

    @abstractmethod
    def process(self, instruction: Instruction) -> tuple[Input, ...]:
        raise NotImplementedError


@dataclass
class EvalInterpreter(Interpreter):
    def process(self, instruction: EvalInstruction) -> tuple[Array, ...]:  # type: ignore
        return instruction.primitive.evaluate(*instruction.inputs, **instruction.params)


@dataclass(frozen=True)
class InterpreterContext:
    stack: tuple[Interpreter, ...] = field(default_factory=lambda: (EvalInterpreter(0),))

    @property
    def key(self) -> tuple[tuple[int, type[Interpreter]], ...]:
        # hashable representation of the stack
        return tuple((interpreter.level, type(interpreter)) for interpreter in self.stack)

    @property
    def interpreter(self) -> Interpreter:
        # top of the stack
        return self.stack[-1]


_interpreter_context_var: ContextVar[InterpreterContext | None] = ContextVar("interpreter_context", default=None)


def interpreter_context() -> InterpreterContext:
    context = _interpreter_context_var.get()
    if context is None:
        context = InterpreterContext()
        _interpreter_context_var.set(context)
    return context


@contextmanager
def interpreter_scope[T: Interpreter](Interpreter: type[T]) -> Iterator[T]:
    # push new interpreter onto the stack
    stack = interpreter_context().stack
    interpreter = Interpreter(level=len(stack))
    token = _interpreter_context_var.set(InterpreterContext(stack=stack + (interpreter,)))

    try:
        yield interpreter
    finally:
        # invalidate interpreter
        interpreter.is_valid = False
        _interpreter_context_var.reset(token)
