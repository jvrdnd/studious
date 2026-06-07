from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from functools import lru_cache, wraps

from .array import Array
from .base import Executable, ExecuteRule, Input, Instruction, Param, Placeholder, PrimitiveLike, Tracer
from .jit import jit
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


def cache(max_size: int = 4096) -> Callable[[ExecuteRule], ExecuteRule]:
    def wrap(function: ExecuteRule) -> ExecuteRule:
        @lru_cache(maxsize=max_size)
        def cached(_: Hashable, primitive: PrimitiveLike, **params: Param) -> Executable:
            return function(primitive, **params)

        @wraps(function)
        def wrapper(primitive: PrimitiveLike, **params: Param) -> Executable:
            return cached(trace_context().key, primitive, **params)

        return wrapper

    return wrap


@cache()
def execute_rule(primitive: PrimitiveLike, **params: Param) -> Executable:
    def function(*inputs: Input) -> tuple[Input, ...]:
        return primitive.bind(*inputs, **params)

    return jit(function)  # type: ignore : input is tracer + params in closure
