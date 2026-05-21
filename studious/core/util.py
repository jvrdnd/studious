from collections.abc import Callable, Hashable
from functools import lru_cache, wraps

from .trace import trace_context


def cache[**P, R](
    max_size: int = 4096,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def wrap(function: Callable[P, R]) -> Callable[P, R]:
        @lru_cache(maxsize=max_size)
        def cached(_: Hashable, *args: P.args, **kwargs: P.kwargs) -> R:
            return function(*args, **kwargs)

        @wraps(function)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            return cached(trace_context().key, *args, **kwargs)

        wrapper.cache_clear = cached.cache_clear  # type: ignore[attr-defined]
        wrapper.cache_info = cached.cache_info  # type: ignore[attr-defined]

        return wrapper

    return wrap
