from collections.abc import Callable
from dataclasses import dataclass, field

from .._lib import Array
from .jit import jit
from .trace import trace_context
from .traceable import Placeholder, Traceable, Tracer
from .util import cache

Executable = Callable[..., Array]
Simulatable = Callable[..., Placeholder]


@dataclass
class Rules:
    execute: Executable | None = None
    simulate: Simulatable | None = None


@dataclass
class Primitive:
    name: str
    rules: Rules = field(default_factory=Rules)

    def execute(self, *args: Array, **kwargs: object) -> Array:
        if self.rules.execute is None:
            raise NotImplementedError(f"no implementation for '{self.name}'")
        return self.rules.execute(*args, **kwargs)

    def simulate(self, *args: Placeholder, **kwargs: object) -> Placeholder:
        if self.rules.simulate is None:
            raise NotImplementedError(f"no implementation for '{self.name}'")
        return self.rules.simulate(*args, **kwargs)

    def bind(self, *args: Traceable, **kwargs: object) -> Traceable:
        # catch leaked tracers
        for arg in args:
            if isinstance(arg, Tracer) and not arg.trace.is_valid:
                raise RuntimeError(f"detected escaped tracer: {arg}")

        return trace_context().trace.process_primitive(self, args, kwargs)


@cache()
def xla_primitive_callable(primitive: Primitive, **kwargs: object) -> Executable:
    def function(*args: Tracer) -> Traceable:
        return primitive.bind(*args, **kwargs)

    jitted = jit(function)
    return jitted
