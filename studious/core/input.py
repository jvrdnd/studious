from dataclasses import dataclass

from .._lib import Dtype
from .array import Array
from .trace import Trace

type TScalar = bool | int | float

# map python types to device types
DTYPES: dict[type[TScalar], Dtype] = {
    bool: Dtype.PRED,
    int: Dtype.S32,
    float: Dtype.F32,
}


# traverses the graph when using a tracer
@dataclass(frozen=True)
class Placeholder:
    dtype: Dtype
    shape: tuple[int, ...]


@dataclass(frozen=True)
class Tracer:
    placeholder: Placeholder
    trace: Trace


# inputs are dynamic values
type Input = TScalar | Array | Tracer

# params are static settings
type Param = TScalar | str | tuple[Param, ...] | None


def get_placeholder(input: Input) -> Placeholder:
    if isinstance(input, Tracer):
        return input.placeholder

    if isinstance(input, Array):
        return Placeholder(input.dtype, tuple(input.shape))

    return Placeholder(DTYPES[type(input)], ())
