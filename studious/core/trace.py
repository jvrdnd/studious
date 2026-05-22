from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import TracebackType

from .._lib import Array
from .input import Input, Param, Tracer, get_placeholder
from .primitive import Primitive

type OpInputs = tuple[Input, ...]
type OpParams = dict[str, Param]
type Op = tuple[Primitive, OpInputs, OpParams]


@dataclass
class Trace:
    level: int
    is_valid: bool = True

    def process(self, primitive: Primitive, inputs: OpInputs, params: OpParams) -> tuple[Input, ...]:
        raise NotImplementedError


@dataclass
class EvalTrace(Trace):
    def process(self, primitive: Primitive, inputs: OpInputs, params: OpParams) -> tuple[Array, ...]:
        return primitive.execute(*inputs, **params)  # type: ignore : called with tuple[Array, ...] inputs


@dataclass
class JitTrace(Trace):
    ops: list[Op] = field(default_factory=lambda: [])

    def process(self, primitive: Primitive, inputs: OpInputs, params: OpParams) -> tuple[Tracer, ...]:
        self.ops.append((primitive, inputs, params))
        in_placeholders = [get_placeholder(input) for input in inputs]
        out_placeholders = primitive.trace(*in_placeholders, **params)
        return tuple(Tracer(placeholder, self) for placeholder in out_placeholders)


@dataclass(frozen=True)
class TraceContext:
    stack: tuple[Trace, ...] = field(default_factory=lambda: (EvalTrace(0),))

    @property
    def key(self) -> tuple[tuple[int, type[Trace]], ...]:
        # hashable representation of the stack
        return tuple((trace.level, type(trace)) for trace in self.stack)

    @property
    def trace(self) -> Trace:
        # top of the stack
        return self.stack[-1]


_trace_context_var: ContextVar[TraceContext | None] = ContextVar("trace_context", default=None)


def trace_context() -> TraceContext:
    context = _trace_context_var.get()
    if context is None:
        context = TraceContext()
        _trace_context_var.set(context)
    return context


@dataclass
class TraceScope[T: Trace]:
    Trace: type[T]
    trace: T = field(init=False)

    token: Token[TraceContext | None] = field(init=False)

    def __enter__(self) -> T:
        # push new trace onto the stack
        stack = trace_context().stack
        self.trace = self.Trace(level=len(stack))
        self.token = _trace_context_var.set(TraceContext(stack=stack + (self.trace,)))
        return self.trace

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ):
        # invalid trace
        self.trace.is_valid = False
        _trace_context_var.reset(self.token)
