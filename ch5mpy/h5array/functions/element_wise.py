from __future__ import annotations

from functools import partial
from numbers import Number
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import numpy.typing as npt
from numpy import _NoValue as NoValue  # type: ignore[attr-defined]

from ch5mpy._typing import NP_FUNC
from ch5mpy.h5array.functions.apply import ApplyOperation, apply, apply_everywhere
from ch5mpy.h5array.functions.implement import implements, register

if TYPE_CHECKING:
    from ch5mpy import H5Array


# ufuncs ----------------------------------------------------------------------
def H5_ufunc(
    a: H5Array[Any],
    out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
    *,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
    dtype: npt.DTypeLike | None = None,
    np_ufunc: NP_FUNC,
) -> Any:
    return apply(
        partial(np_ufunc, dtype=dtype),
        ApplyOperation.set,
        a,
        out=None if out is None else out[0],
        dtype=dtype,
        initial=NoValue,
        where=where,
        default_0D_output=False,
    )


_IMPLEMENTED_UFUNCS: tuple[NP_FUNC, ...] = (
    np.sin,
    np.cos,
    np.tan,
    np.arcsin,
    np.arccos,
    np.arctan,
    np.sinh,
    np.cosh,
    np.tanh,
    np.arcsinh,
    np.arccosh,
    np.arctanh,
    np.floor,
    np.ceil,
    np.trunc,
    np.exp,
    np.expm1,
    np.exp2,
    np.log,
    np.log10,
    np.log2,
    np.log1p,
    np.positive,
    np.negative,
    np.sqrt,
    np.cbrt,
    np.square,
    np.absolute,
    np.fabs,
    np.sign,
)

for ufunc in _IMPLEMENTED_UFUNCS:
    register(partial(H5_ufunc, np_ufunc=ufunc), ufunc)


@implements(np.isfinite)
def isfinite(
    a: H5Array[Any],
    out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
    *,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> Any:
    return apply(
        partial(np.isfinite),
        ApplyOperation.set,
        a,
        out=None if out is None else out[0],
        dtype=bool,
        initial=NoValue,
        where=where,
        default_0D_output=False,
    )


@implements(np.isinf)
def isinf(
    a: H5Array[Any],
    out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
    *,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> Any:
    return apply(
        partial(np.isinf),
        ApplyOperation.set,
        a,
        out=None if out is None else out[0],
        dtype=bool,
        initial=NoValue,
        where=where,
        default_0D_output=False,
    )


@implements(np.isnan)
def isnan(
    a: H5Array[Any],
    out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None,
    *,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> Any:
    return apply(
        partial(np.isnan),
        ApplyOperation.set,
        a,
        out=None if out is None else out[0],
        dtype=bool,
        initial=NoValue,
        where=where,
        default_0D_output=False,
    )


@implements(np.isneginf)
def isneginf(a: H5Array[Any], out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None) -> Any:
    return apply_everywhere(
        partial(np.isneginf),
        ApplyOperation.set,
        a,
        out=None if out is None else out[0],
        dtype=bool,
        initial=NoValue,
        default_0D_output=False,
    )


@implements(np.isposinf)
def isposinf(a: H5Array[Any], out: tuple[H5Array[Any] | npt.NDArray[Any]] | None = None) -> Any:
    return apply_everywhere(
        partial(np.isposinf),
        ApplyOperation.set,
        a,
        out=None if out is None else out[0],
        dtype=bool,
        initial=NoValue,
        default_0D_output=False,
    )


# numpy functions -------------------------------------------------------------
@implements(np.prod)
def prod(
    a: H5Array[Any],
    axis: int | Iterable[int] | tuple[int] | None = None,
    dtype: npt.DTypeLike | None = None,
    out: H5Array[Any] | npt.NDArray[Any] | None = None,
    keepdims: bool = False,
    initial: int | float | complex | NoValue = NoValue,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> Any:
    initial = 1 if initial is NoValue else initial

    return apply(
        partial(np.prod, keepdims=keepdims, dtype=dtype, axis=axis),
        ApplyOperation.imul,
        a,
        out,
        dtype=dtype,
        initial=initial,
        where=where,
    )


@implements(np.sum)
def sum_(
    a: H5Array[Any],
    axis: int | Iterable[int] | tuple[int] | None = None,
    dtype: npt.DTypeLike | None = None,
    out: H5Array[Any] | npt.NDArray[Any] | None = None,
    keepdims: bool = False,
    initial: int | float | complex | NoValue = NoValue,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> Any:
    initial = 0 if initial is NoValue else initial

    return apply(
        partial(np.sum, keepdims=keepdims, dtype=dtype, axis=axis),
        ApplyOperation.iadd,
        a,
        out,
        dtype=dtype,
        initial=initial,
        where=where,
    )


@implements(np.mean)
def mean(
    a: H5Array[Any],
    axis: int | Iterable[int] | tuple[int] | None = None,
    dtype: npt.DTypeLike | None = None,
    out: H5Array[Any] | npt.NDArray[Any] | None = None,
    keepdims: bool = False,
    *,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> Any:
    s = sum_(a, axis, dtype, out, keepdims, where=where)

    if where == NoValue:
        where = True

    n = np.broadcast_to(where, a.shape).sum(axis=axis)  # type: ignore[arg-type]
    return np.divide(s, n, out, where=where)  # type: ignore[arg-type]


@implements(np.amax)
def amax(
    a: H5Array[Any],
    axis: int | Iterable[Any] | tuple[int] | None = None,
    out: H5Array[Any] | npt.NDArray[Any] | None = None,
    keepdims: bool = False,
    initial: Number | NoValue = NoValue,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> npt.NDArray[np.number[Any]] | np.number[Any]:
    return apply(  # type: ignore[no-any-return]
        partial(np.amax, keepdims=keepdims, axis=axis),
        ApplyOperation.set,
        a,
        out,
        dtype=None,
        initial=initial,
        where=where,
    )


@implements(np.amin)
def amin(
    a: H5Array[Any],
    axis: int | Iterable[Any] | tuple[int] | None = None,
    out: H5Array[Any] | npt.NDArray[Any] | None = None,
    keepdims: bool = False,
    initial: Number | NoValue = NoValue,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> npt.NDArray[np.number[Any]] | np.number[Any]:
    return apply(  # type: ignore[no-any-return]
        partial(np.amin, keepdims=keepdims, axis=axis),
        ApplyOperation.set,
        a,
        out,
        dtype=None,
        initial=initial,
        where=where,
    )


@implements(np.all)
def all_(
    a: H5Array[Any],
    axis: int | Iterable[Any] | tuple[int] | None = None,
    out: H5Array[Any] | npt.NDArray[Any] | None = None,
    keepdims: bool = False,
    *,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> npt.NDArray[Any] | bool:
    return apply(  # type: ignore[no-any-return]
        partial(np.all, keepdims=keepdims, axis=axis),
        ApplyOperation.iand,
        a,
        out,
        dtype=bool,
        initial=True,
        where=where,
    )


@implements(np.any)
def any_(
    a: H5Array[Any],
    axis: int | Iterable[Any] | tuple[int] | None = None,
    out: H5Array[Any] | npt.NDArray[Any] | None = None,
    keepdims: bool = False,
    *,
    where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue,
) -> npt.NDArray[Any] | bool:
    return apply(  # type: ignore[no-any-return]
        partial(np.any, keepdims=keepdims, axis=axis),
        ApplyOperation.ior,
        a,
        out,
        dtype=bool,
        initial=False,
        where=where,
    )
