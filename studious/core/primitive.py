from dataclasses import dataclass

from .._lib import Array
from .trace import Traceable, TraceContext, Tracer

context = TraceContext()


@dataclass
class Primitive:
    name: str

    def impl(self, *args: Array, **params: object) -> Array:
        raise NotImplementedError(f"no implementation for '{self.name}'")

    def bind(self, *args: Traceable, **params: object) -> Traceable:
        # catch leaked tracers
        for arg in args:
            if isinstance(arg, Tracer) and not arg.trace.is_valid:
                raise RuntimeError(f"detected escaped tracer: {arg}")

        return context.trace.process_primitive(self, args, params)
