from collections.abc import Callable, Hashable
from functools import lru_cache, wraps

from .input import Param
from .jit import Executable
from .primitive import ExecuteRule, Primitive
from .trace import trace_context


def cache(max_size: int = 4096) -> Callable[[ExecuteRule], ExecuteRule]:
    def wrap(function: ExecuteRule) -> ExecuteRule:
        @lru_cache(maxsize=max_size)
        def cached(_: Hashable, primitive: Primitive, **params: Param) -> Executable:
            return function(primitive, **params)

        @wraps(function)
        def wrapper(primitive: Primitive, **params: Param) -> Executable:
            return cached(trace_context().key, primitive, **params)

        return wrapper

    return wrap
