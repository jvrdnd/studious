from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from functools import lru_cache, wraps

from .array import Array
from .trace import Trace, TraceScope, trace_context
from .types import (
    Executable,
    ExecuteRule,
    Input,
    Instruction,
    Param,
    Placeholder,
    PrimitiveLike,
    Traceable,
    Tracer,
)


@dataclass
class JitTrace(Trace):
    instructions: list[JitTracerInstruction] = field(default_factory=lambda: [])

    def array_to_level(self, array: Array) -> JitTracer:
        return JitTracer(self, Placeholder(array.dtype, array.shape))

    def tracer_to_level(self, tracer: Tracer) -> JitTracer:
        return JitTracer(self, tracer.placeholder)

    def process(self, instruction: JitTracerInstruction) -> tuple[Tracer, ...]:  # type: ignore
        self.instructions.append(instruction)
        in_placeholders = [input.placeholder for input in instruction.inputs]
        out_placeholders = instruction.primitive.trace(*in_placeholders, **instruction.params)
        return tuple(JitTracer(self, placeholder) for placeholder in out_placeholders)


@dataclass
class JitTracer(Tracer):
    _trace: JitTrace
    _placeholder: Placeholder

    @property
    def trace(self) -> JitTrace:
        return self._trace

    @property
    def placeholder(self) -> Placeholder:
        return self._placeholder


@dataclass(frozen=True)
class JitTracerInstruction(Instruction):
    inputs: tuple[JitTracer, ...]


def compile(instructions: list[JitTracerInstruction], tracers: tuple[Tracer, ...]) -> Executable:
    raise NotImplementedError


def jit(function: Traceable) -> Executable:
    @wraps(function)
    def wrapper(*inputs: Array, **params: Param) -> tuple[Array, ...]:
        # push the jit trace
        with TraceScope(JitTrace) as trace:
            # gather ops
            in_tracers = tuple(JitTracer(trace, Placeholder(input.dtype, input.shape)) for input in inputs)
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
