from __future__ import annotations

from functools import wraps

from .array import Array
from .base import Executable, Param, Placeholder, Traceable, Tracer, TracerInstruction
from .trace import JitTrace, TraceScope


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
