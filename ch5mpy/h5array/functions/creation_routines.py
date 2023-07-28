from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import Literal
from typing import TypeVar
from typing import Iterable
from typing import TYPE_CHECKING

from ch5mpy.h5array.functions.implement import implements

if TYPE_CHECKING:
    from ch5mpy import H5Array


_T = TypeVar("_T", bound=np.generic)


@implements(np.copy)
def copy(a: H5Array[_T]) -> npt.NDArray[_T]:
    return np.array(a)


@implements(np.zeros_like)
def zeros_like(
    a: H5Array[Any],
    dtype: npt.DTypeLike | None = None,
    order: Literal["C", "F"] | None = "C",
    shape: Iterable[int] | None = None,
) -> Any:
    return np.zeros(shape=shape or a.shape, dtype=dtype or a.dtype, order=order)  # type: ignore[call-overload]


@implements(np.ones_like)
def ones_like(
    a: H5Array[Any],
    dtype: npt.DTypeLike = None,
    order: Literal["C", "F"] | None = "C",
    shape: Iterable[int] | None = None,
) -> Any:
    return np.ones(shape=shape or a.shape, dtype=dtype or a.dtype, order=order)  # type: ignore[call-overload]
