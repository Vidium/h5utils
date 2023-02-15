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
from h5utils.h5array.functions.apply import apply_2
from h5utils.h5array.functions.implement import implements


if TYPE_CHECKING:
    from h5utils import H5Array


# ====================================================
# code
def _cast_H5Array(obj: Any) -> H5Array[Any]:
    return obj                                                                             # type: ignore[no-any-return]


def ensure_h5array_first(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                         x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any]) \
        -> tuple[H5Array[Any], npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any]]:
    if not isinstance(x1, h5utils.H5Array):
        return _cast_H5Array(x2), x1

    return _cast_H5Array(x1), x2


@implements(np.array_equal)
def array_equal(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                equal_nan: bool = False) -> bool:
    x1, x2 = ensure_h5array_first(x1, x2)

    # case 0D
    if isinstance(x2, Number):
        if x1.ndim:
            return False

        return np.array_equal(x1, x2, equal_nan=equal_nan)                                  # type: ignore[arg-type]

    # case nD
    if not isinstance(x2, (np.ndarray, h5utils.H5Array)):
        x2 = np.array(x2)

    if x1.shape != x2.shape:
        return False

    for index, chunk_x1, chunk_x2 in iter_chunks_2(x1, x2):
        if not np.array_equal(chunk_x1, chunk_x2):                                              # type: ignore[arg-type]
            return False

    return True


@implements(np.greater)
def greater(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
            x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
            out: H5Array[Any] | npt.NDArray[Any] | None = None,
            where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
            dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.greater, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=False)


@implements(np.greater_equal)
def greater_equal(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                  x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                  out: H5Array[Any] | npt.NDArray[Any] | None = None,
                  where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
                  dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.greater_equal, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=False)


@implements(np.less)
def less(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
         x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
         dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.less, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=False)


@implements(np.less_equal)
def less_equal(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
               x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
               out: H5Array[Any] | npt.NDArray[Any] | None = None,
               where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
               dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.less_equal, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=False)


@implements(np.equal)
def equal(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
          x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
          dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.equal, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=False)


@implements(np.not_equal)
def not_equal(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
              x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
              out: H5Array[Any] | npt.NDArray[Any] | None = None,
              where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
              dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.not_equal, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=False)


@implements(np.multiply)
def multiply(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
             x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
             out: H5Array[Any] | npt.NDArray[Any] | None = None,
             where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
             dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.multiply, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)
