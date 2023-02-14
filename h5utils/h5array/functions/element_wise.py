# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from numpy import _NoValue as NoValue                                                       # type: ignore[attr-defined]
from functools import partial

import numpy.typing as npt
from typing import Any
from typing import Iterable
from typing import TYPE_CHECKING

from h5utils.h5array.functions.implement import implements
from h5utils.h5array.functions.apply import apply

if TYPE_CHECKING:
    from h5utils import H5Array


# ====================================================
# code
@implements(np.sum)
def sum(a: H5Array[Any],
        axis: int | Iterable[int] | tuple[int] | None = None,
        dtype: npt.DTypeLike | None = None,
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        keepdims: bool = False,
        initial: int | float | complex | None = None,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    return apply(partial(np.sum, keepdims=keepdims, dtype=dtype, axis=axis), '__iadd__', a, out,
                 dtype=dtype, initial=initial, where=where)


@implements(np.all)
def all(a: H5Array[Any],
        axis: int | Iterable[Any] | tuple[int] | None = None,
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        keepdims: bool = False,
        *,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> npt.NDArray[Any] | bool:
    return apply(partial(np.all, keepdims=keepdims, axis=axis), '__iand__', a, out,  # type: ignore[no-any-return]
                 dtype=bool, initial=True, where=where)


@implements(np.any)
def any(a: H5Array[Any],
        axis: int | Iterable[Any] | tuple[int] | None = None,
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        keepdims: bool = False,
        *,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> npt.NDArray[Any] | bool:
    return apply(partial(np.any, keepdims=keepdims, axis=axis), '__ior__', a, out,  # type: ignore[no-any-return]
                 dtype=bool, initial=False, where=where)
