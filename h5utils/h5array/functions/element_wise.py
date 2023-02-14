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
from h5utils._typing import NP_FUNC

if TYPE_CHECKING:
    from h5utils import H5Array


# ====================================================
# code
def _elementwise_add_from_zero(a: H5Array[Any],
                               out: H5Array[Any] | npt.NDArray[Any] | None,
                               func: NP_FUNC | np.ufunc,
                               where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue,
                               dtype: npt.DTypeLike | None) -> Any:
    return apply(partial(func, dtype=dtype), '__iadd__', a, out, dtype=dtype, initial=0, where=where)


@implements(np.floor)
def floor(a: H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.floor, where, dtype)


@implements(np.ceil)
def ceil(a: H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.ceil, where, dtype)


@implements(np.trunc)
def trunc(a: H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.trunc, where, dtype)


@implements(np.prod)
def prod(a: H5Array[Any],
         axis: int | Iterable[int] | tuple[int] | None = None,
         dtype: npt.DTypeLike | None = None,
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         keepdims: bool = False,
         initial: int | float | complex | None = None,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    return apply(partial(np.prod, keepdims=keepdims, dtype=dtype, axis=axis), '__imul__', a, out,
                 dtype=dtype, initial=1, where=where)


@implements(np.sum)
def sum(a: H5Array[Any],
        axis: int | Iterable[int] | tuple[int] | None = None,
        dtype: npt.DTypeLike | None = None,
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        keepdims: bool = False,
        initial: int | float | complex | None = None,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    initial = 0 if initial is None else initial

    return apply(partial(np.sum, keepdims=keepdims, dtype=dtype, axis=axis), '__iadd__', a, out,
                 dtype=dtype, initial=initial, where=where)


@implements(np.exp)
def exp(a: H5Array[Any],
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        *,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
        dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.exp, where, dtype)


@implements(np.expm1)
def expm1(a: H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.expm1, where, dtype)


@implements(np.exp2)
def exp2(a: H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.exp2, where, dtype)


@implements(np.log)
def log(a: H5Array[Any],
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        *,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
        dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.log, where, dtype)


@implements(np.log10)
def log10(a: H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.log10, where, dtype)


@implements(np.log2)
def log2(a: H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.log2, where, dtype)


@implements(np.log1p)
def log1p(a: H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.log1p, where, dtype)


@implements(np.positive)
def positive(a: H5Array[Any],
             out: H5Array[Any] | npt.NDArray[Any] | None = None,
             *,
             where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
             dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.positive, where, dtype)


@implements(np.negative)
def negative(a: H5Array[Any],
             out: H5Array[Any] | npt.NDArray[Any] | None = None,
             *,
             where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
             dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.negative, where, dtype)


@implements(np.sqrt)
def sqrt(a: H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.sqrt, where, dtype)


@implements(np.cbrt)
def cbrt(a: H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.cbrt, where, dtype)


@implements(np.square)
def square(a: H5Array[Any],
           out: H5Array[Any] | npt.NDArray[Any] | None = None,
           *,
           where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
           dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.square, where, dtype)


@implements(np.absolute)
def absolute(a: H5Array[Any],
             out: H5Array[Any] | npt.NDArray[Any] | None = None,
             *,
             where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
             dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.absolute, where, dtype)


@implements(np.fabs)
def fabs(a: H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.fabs, where, dtype)


@implements(np.sign)
def sign(a: H5Array[Any],
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _elementwise_add_from_zero(a, out, np.sign, where, dtype)


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


@implements(np.isfinite)
def isfinite(a: H5Array[Any],
             out: H5Array[Any] | npt.NDArray[Any] | None = None,
             *,
             where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    return apply(partial(np.any), '__ior__', a, out, dtype=bool, initial=False, where=where)


@implements(np.isinf)
def isinf(a: H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    return apply(partial(np.isinf), '__ior__', a, out, dtype=bool, initial=False, where=where)


@implements(np.isnan)
def isnan(a: H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    return apply(partial(np.isnan), '__ior__', a, out, dtype=bool, initial=False, where=where)


@implements(np.isneginf)
def isneginf(a: H5Array[Any],
             out: H5Array[Any] | npt.NDArray[Any] | None = None) -> Any:
    return apply(partial(np.isneginf), '__ior__', a, out, dtype=bool, initial=False, where=NoValue)


@implements(np.isposinf)
def isposinf(a: H5Array[Any],
             out: H5Array[Any] | npt.NDArray[Any] | None = None) -> Any:
    return apply(partial(np.isposinf), '__ior__', a, out, dtype=bool, initial=False, where=NoValue)
