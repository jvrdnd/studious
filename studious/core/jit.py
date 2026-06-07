from __future__ import annotations

from collections.abc import Callable, Hashable
from functools import lru_cache, wraps

from .array import Array
from .base import (
    Executable,
    ExecuteRule,
    Input,
    Param,
    Placeholder,
    PrimitiveLike,
    Traceable,
    Tracer,
    TracerInstruction,
)
from .trace import JitTrace, TraceScope, trace_context


def compile(instructions: list[TracerInstruction], tracers: tuple[Tracer, ...]) -> Executable:
    raise NotImplementedError


def jit(function: Traceable) -> Executable:
    @wraps(function)
    def wrapper(*inputs: Array, **params: Param) -> tuple[Array, ...]:
        # push the jit trace
        with TraceScope(JitTrace) as trace:
            # gather ops
            in_tracers = [Tracer(Placeholder(input.dtype, input.shape), trace) for input in inputs]
            out_tracers = function(*in_tracers, **params)

        return compile(trace.instructions, out_tracers)(*inputs)

    return wrapper


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
