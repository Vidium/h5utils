# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from numbers import Number

import numpy.typing as npt
from typing import Any
from typing import Iterable
from typing import TYPE_CHECKING

import h5utils
from h5utils.h5array.inplace import iter_chunks_2
from h5utils.h5array.functions.implement import implements


if TYPE_CHECKING:
    from h5utils import H5Array


# ====================================================
# code
def _cast_H5Array(obj: Any) -> H5Array[Any]:
    return obj                                                                             # type: ignore[no-any-return]


@implements(np.array_equal)
def array_equal(x1: npt.NDArray[Any] | Iterable[Any] | int | float | H5Array[Any],
                x2: npt.NDArray[Any] | Iterable[Any] | int | float | H5Array[Any],
                equal_nan: bool = False) -> bool:
    # ensure x1 is an H5Array
    if not isinstance(x1, h5utils.H5Array):
        x1, x2 = _cast_H5Array(x2), x1

    # case 0D
    if isinstance(x2, Number):
        if x1.ndim:
            return False

        return np.array_equal(x1[()], x2, equal_nan=equal_nan)                                  # type: ignore[arg-type]

    # case nD
    if not isinstance(x2, (np.ndarray, h5utils.H5Array)):
        x2 = np.array(x2)

    if x1.shape != x2.shape:
        return False

    for index, chunk_x1, chunk_x2 in iter_chunks_2(x1, x2):
        if not np.array_equal(chunk_x1, chunk_x2):
            return False

    return True
