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

import ch5mpy
from ch5mpy.h5array.inplace import iter_chunks_2
from ch5mpy.h5array.functions.apply import apply_2
from ch5mpy.h5array.functions.implement import implements


if TYPE_CHECKING:
    from ch5mpy import H5Array


# ====================================================
# code
def _cast_H5Array(obj: Any) -> H5Array[Any]:
    return obj                                                                             # type: ignore[no-any-return]


def ensure_h5array_first(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                         x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any]) \
        -> tuple[H5Array[Any], npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any]]:
    if not isinstance(x1, ch5mpy.H5Array):
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
    if not isinstance(x2, (np.ndarray, ch5mpy.H5Array)):
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


@implements(np.add)
def add(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
        x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
        dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.add, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.multiply)
def multiply(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
             x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
             out: H5Array[Any] | npt.NDArray[Any] | None = None,
             where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
             dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.multiply, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.divide)
def divide(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
           x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
           out: H5Array[Any] | npt.NDArray[Any] | None = None,
           where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
           dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.divide, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.power)
def power(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
          x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
          dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.power, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.subtract)
def subtract(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
             x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
             out: H5Array[Any] | npt.NDArray[Any] | None = None,
             where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
             dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.subtract, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.true_divide)
def true_divide(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                out: H5Array[Any] | npt.NDArray[Any] | None = None,
                where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
                dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.true_divide, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.floor_divide)
def floor_divide(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                 x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                 out: H5Array[Any] | npt.NDArray[Any] | None = None,
                 where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
                 dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.floor_divide, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.float_power)
def float_power(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                out: H5Array[Any] | npt.NDArray[Any] | None = None,
                where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
                dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.float_power, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.fmod)
def fmod(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
         x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
         dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.fmod, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.mod)
def mod(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
        x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
        dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.mod, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.maximum)
def maximum(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
            x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
            out: H5Array[Any] | npt.NDArray[Any] | None = None,
            where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
            dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.maximum, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.fmax)
def fmax(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
         x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
         dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.fmax, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.minimum)
def minimum(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
            x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
            out: H5Array[Any] | npt.NDArray[Any] | None = None,
            where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
            dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.minimum, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.fmin)
def fmin(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
         x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
         dtype: npt.DTypeLike | None = None) -> Any:
    return apply_2(np.fmin, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.logical_and)
def logical_and(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                out: H5Array[Any] | npt.NDArray[Any] | None = None,
                where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
                dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.logical_and, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.logical_or)
def logical_or(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
               x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
               out: H5Array[Any] | npt.NDArray[Any] | None = None,
               where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
               dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.logical_or, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.logical_not)
def logical_not(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                out: H5Array[Any] | npt.NDArray[Any] | None = None,
                where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
                dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.logical_not, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)


@implements(np.logical_xor)
def logical_xor(x1: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                x2: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
                out: H5Array[Any] | npt.NDArray[Any] | None = None,
                where: npt.NDArray[np.bool_] | Iterable[bool] | bool = True,
                dtype: npt.DTypeLike | None = None) -> Any:
    if dtype is None:
        dtype = bool

    return apply_2(np.logical_xor, *ensure_h5array_first(x1, x2), out=out, dtype=dtype, where=where, default=x1)
