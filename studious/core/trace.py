from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import TracebackType

from .._lib import Array
from .primitive import Primitive
from .traceable import Traceable, Tracer, get_placeholder


@dataclass
class Trace:
    level: int
    is_valid: bool = True

    def process_primitive(
        self, primitive: Primitive, args: tuple[Traceable, ...], params: dict[str, object]
    ) -> Traceable:
        raise NotImplementedError


@dataclass
class EvalTrace(Trace):
    def process_primitive(self, primitive: Primitive, args: tuple[Traceable, ...], params: dict[str, object]) -> Array:
        return primitive.execute(*args, **params)  # type: ignore[arg-type]: called with tuple[Array, ...] args


@dataclass
class JitTrace(Trace):
    ops: list[tuple[Primitive, tuple[Traceable, ...], dict[str, object]]] = field(default_factory=lambda: [])

    def process_primitive(self, primitive: Primitive, args: tuple[Traceable, ...], params: dict[str, object]) -> Tracer:
        self.ops.append((primitive, args, params))
        in_placeholders = [get_placeholder(arg) for arg in args]
        out_placeholder = primitive.simulate(*in_placeholders, **params)
        return Tracer(out_placeholder, self)


@dataclass(frozen=True)
class TraceContext:
    stack: tuple[Trace, ...] = field(default_factory=lambda: (EvalTrace(0),))

    @property
    def key(self) -> tuple[tuple[int, type[Trace]], ...]:
        # hashable representation of the stack
        return tuple((trace.level, type(trace)) for trace in self.stack)

    @property
    def trace(self) -> Trace:
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
