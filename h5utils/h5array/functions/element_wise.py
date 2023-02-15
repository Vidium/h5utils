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
# ufuncs ----------------------------------------------------------------------
def _apply_ufunc(a: H5Array[Any],
                 out: tuple[H5Array[Any] | npt.NDArray[Any]] | None,
                 ufunc: partial[np.ufunc],
                 where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue,
                 dtype: npt.DTypeLike | None) -> Any:
    return apply(ufunc, '__set__', a,
                 out=None if out is None else out[0],
                 dtype=dtype,
                 initial=None,
                 where=where,
                 default_0D_output=False)


@implements(np.floor)
def floor(a: H5Array[Any],
          out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.floor, dtype=dtype), where, dtype)


@implements(np.ceil)
def ceil(a: H5Array[Any],
         out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.ceil, dtype=dtype), where, dtype)


@implements(np.trunc)
def trunc(a: H5Array[Any],
          out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.trunc, dtype=dtype), where, dtype)


@implements(np.exp)
def exp(a: H5Array[Any],
        out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
        *,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
        dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.exp, dtype=dtype), where, dtype)


@implements(np.expm1)
def expm1(a: H5Array[Any],
          out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.expm1, dtype=dtype), where, dtype)


@implements(np.exp2)
def exp2(a: H5Array[Any],
         out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.exp2, dtype=dtype), where, dtype)


@implements(np.log)
def log(a: H5Array[Any],
        out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
        *,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
        dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.log, dtype=dtype), where, dtype)


@implements(np.log10)
def log10(a: H5Array[Any],
          out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.log10, dtype=dtype), where, dtype)


@implements(np.log2)
def log2(a: H5Array[Any],
         out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.log2, dtype=dtype), where, dtype)


@implements(np.log1p)
def log1p(a: H5Array[Any],
          out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
          dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.log1p, dtype=dtype), where, dtype)


@implements(np.positive)
def positive(a: H5Array[Any],
             out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
             *,
             where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
             dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.positive, dtype=dtype), where, dtype)


@implements(np.negative)
def negative(a: H5Array[Any],
             out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
             *,
             where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
             dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.negative, dtype=dtype), where, dtype)


@implements(np.sqrt)
def sqrt(a: H5Array[Any],
         out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.sqrt, dtype=dtype), where, dtype)


@implements(np.cbrt)
def cbrt(a: H5Array[Any],
         out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.cbrt, dtype=dtype), where, dtype)


@implements(np.square)
def square(a: H5Array[Any],
           out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
           *,
           where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
           dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.square, dtype=dtype), where, dtype)


@implements(np.absolute)
def absolute(a: H5Array[Any],
             out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
             *,
             where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
             dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.absolute, dtype=dtype), where, dtype)


@implements(np.fabs)
def fabs(a: H5Array[Any],
         out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.fabs, dtype=dtype), where, dtype)


@implements(np.sign)
def sign(a: H5Array[Any],
         out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
         dtype: npt.DTypeLike | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.sign, dtype=dtype), where, dtype)


@implements(np.isfinite)
def isfinite(a: H5Array[Any],
             out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
             *,
             where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    return _apply_ufunc(a, out, partial(np.isfinite), where, dtype=bool)


@implements(np.isinf)
def isinf(a: H5Array[Any],
          out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    return _apply_ufunc(a, out, partial(np.isinf), where, dtype=bool)


@implements(np.isnan)
def isnan(a: H5Array[Any],
          out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    return _apply_ufunc(a, out, partial(np.isnan), where, dtype=bool)


@implements(np.isneginf)
def isneginf(a: H5Array[Any],
             out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.isneginf), where=NoValue, dtype=bool)


@implements(np.isposinf)
def isposinf(a: H5Array[Any],
             out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None) -> Any:
    return _apply_ufunc(a, out, partial(np.isposinf), where=NoValue, dtype=bool)


# numpy functions -------------------------------------------------------------
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
