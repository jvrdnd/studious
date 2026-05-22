from dataclasses import dataclass

from .._lib import Array, DType
from .trace import Trace

type TScalar = bool | int | float

# map python types to device types
DTYPES: dict[type[TScalar], DType] = {
    bool: DType.BOOL,
    int: DType.INT32,
    float: DType.FLOAT32,
}


# traverses the graph when using a tracer
@dataclass(frozen=True)
class Placeholder:
    dtype: DType
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
