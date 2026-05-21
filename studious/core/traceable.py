from dataclasses import dataclass

from .._lib import Array, DType
from .trace import Trace

DTYPES: dict[type, DType] = {
    bool: DType.BOOL,
    int: DType.INT32,
    float: DType.FLOAT32,
}


@dataclass
class Placeholder:
    dtype: DType
    shape: tuple[int, ...]


@dataclass
class Tracer:
    placeholder: Placeholder
    trace: Trace


Scalar = (bool, float, int)
Traceable = bool | float | int | Array | Tracer


def get_placeholder(traceable: Traceable) -> Placeholder:
    if isinstance(traceable, Tracer):
        return traceable.placeholder

    if isinstance(traceable, Scalar):
        return Placeholder(DTYPES[type(traceable)], ())

    raise TypeError(f"cannot get placeholder for {type(traceable)}")
