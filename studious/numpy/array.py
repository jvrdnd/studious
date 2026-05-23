from numpy import bool_, float32, float64, int32, int64
from numpy.typing import NDArray

from .._lib import Array, from_numpy


def array[NPArray: (NDArray[bool_], NDArray[int32], NDArray[int64], NDArray[float32], NDArray[float64])](
    array: NPArray,
) -> Array:
    return from_numpy(array)
