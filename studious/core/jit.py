from __future__ import annotations

from functools import wraps
from typing import Protocol

from .array import Array
from .input import Input, Param, Tracer, get_placeholder
from .trace import JitTrace, Op, TraceScope


class Traceable(Protocol):
    def __call__(self, *inputs: Input, **params: Param) -> tuple[Tracer, ...]: ...


class Executable(Protocol):
    def __call__(self, *inputs: Input) -> tuple[Array, ...]: ...


def compile(ops: list[Op], tracers: tuple[Tracer, ...]) -> Executable:
    raise NotImplementedError


def jit(function: Traceable) -> Executable:
    @wraps(function)
    def wrapper(*inputs: Input, **params: Param) -> tuple[Array, ...]:
        # push the jit trace
        with TraceScope(JitTrace) as trace:
            # gather ops
            in_tracers = [Tracer(get_placeholder(input), trace) for input in inputs]
            out_tracers = function(*in_tracers, **params)

        return compile(trace.ops, out_tracers)(*inputs)

    return wrapper
