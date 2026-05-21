from collections.abc import Callable
from functools import wraps

from .primitive import Executable
from .trace import JitTrace, TraceScope


def jit[**P, R](function: Callable[P, R]) -> Callable[P, Executable]:
    @wraps(function)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> Executable:
        # push the JIT trace
        with TraceScope(JitTrace) as trace:
            # traverse
            function(*args, **kwargs)
            graph = trace.ops

        # compile
        return graph  # type: ignore[arg-type]

    return wrapper
