from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, Protocol, cast

import numpy as np
import numpy.typing as npt

import ch5mpy
from ch5mpy.write import _store_dataset


_NoValue = object()


class ArrayCreationFunc:
    # region magic methods
    def __init__(self, name: str) -> None:
        self._fill_value: Any = _NoValue
        self._name = name

    def __repr__(self) -> str:
        return f"<function ch5mpy.{self._name} at {hex(id(self))}>"

    def __call__(
        self,
        shape: int | tuple[int, ...],
        name: str,
        loc: str | Path | ch5mpy.File | ch5mpy.Group,
        dtype: npt.DTypeLike = np.float64,
        chunks: bool | tuple[int, ...] = True,
        maxshape: int | tuple[int | None, ...] | None = None,
    ) -> ch5mpy.H5Array[Any]:
        if self._fill_value == _NoValue:
            raise RuntimeError

        shape = shape if isinstance(shape, tuple) else (shape,)

        if not isinstance(loc, (ch5mpy.File, ch5mpy.Group)):
            loc = ch5mpy.File(loc, mode=ch5mpy.H5Mode.READ_WRITE_CREATE)

        dset = _store_dataset(
            loc, name, shape=shape, dtype=dtype, chunks=chunks, maxshape=maxshape, fill_value=self._fill_value
        )

        return ch5mpy.H5Array(dset)

    # endregion

    # region methods
    def p(
        self,
        shape: int | tuple[int, ...],
        dtype: npt.DTypeLike = np.float64,
        chunks: bool | tuple[int, ...] = True,
        maxshape: int | tuple[int | None, ...] | None = None,
    ) -> partial[ch5mpy.H5Array[Any]]:
        return partial(self.__call__, shape=shape, dtype=dtype, chunks=chunks, maxshape=maxshape)

    # endregion


class ArrayCreationFuncWithFill(ArrayCreationFunc):
    # region magic methods
    def __init__(self, name: str, fill_value: Any):
        super().__init__(name)
        self._fill_value = fill_value

    def __call__(
        self,
        shape: int | tuple[int, ...],
        name: str,
        loc: str | Path | ch5mpy.File | ch5mpy.Group,
        dtype: npt.DTypeLike = np.float64,
        chunks: bool | tuple[int, ...] = True,
        maxshape: int | tuple[int | None, ...] | None = None,
    ) -> ch5mpy.H5Array[Any]:
        return super().__call__(shape, name, loc, dtype, chunks, maxshape)

    # endregion


class _ArrayCreationFuncProtocol(Protocol):
    acf: ArrayCreationFunc

    def __call__(
        self,
        shape: int | tuple[int, ...],
        name: str,
        loc: str | Path | ch5mpy.File | ch5mpy.Group,
        dtype: npt.DTypeLike = np.float64,
        chunks: bool | tuple[int, ...] = True,
        maxshape: int | tuple[int | None, ...] | None = None,
    ) -> ch5mpy.H5Array[Any]:
        ...

    def p(
        self,
        shape: int | tuple[int, ...],
        dtype: npt.DTypeLike = np.float64,
        chunks: bool | tuple[int, ...] = True,
        maxshape: int | tuple[int | None, ...] | None = None,
    ) -> partial[ch5mpy.H5Array[Any]]:
        ...


def with_partial(acf: ArrayCreationFunc) -> Callable[[Callable[..., Any]], _ArrayCreationFuncProtocol]:
    """Add a `.p()` method to a function for partial calls."""

    def decorator(func: Callable[..., Any]) -> _ArrayCreationFuncProtocol:
        setattr(func, "acf", acf)
        setattr(func, "p", acf.p)
        return cast(_ArrayCreationFuncProtocol, func)

    return decorator


@with_partial(ArrayCreationFuncWithFill("empty", None))
def empty(
    shape: int | tuple[int, ...],
    name: str,
    loc: str | Path | ch5mpy.File | ch5mpy.Group,
    dtype: npt.DTypeLike = np.float64,
    chunks: bool | tuple[int, ...] = True,
    maxshape: int | tuple[int | None, ...] | None = None,
) -> ch5mpy.H5Array[Any]:
    return empty.acf(shape, name, loc, dtype, chunks, maxshape)


@with_partial(ArrayCreationFuncWithFill("zeros", 0))
def zeros(
    shape: int | tuple[int, ...],
    name: str,
    loc: str | Path | ch5mpy.File | ch5mpy.Group,
    dtype: npt.DTypeLike = np.float64,
    chunks: bool | tuple[int, ...] = True,
    maxshape: int | tuple[int | None, ...] | None = None,
) -> ch5mpy.H5Array[Any]:
    return zeros.acf(shape, name, loc, dtype, chunks, maxshape)


@with_partial(ArrayCreationFuncWithFill("ones", 1))
def ones(
    shape: int | tuple[int, ...],
    name: str,
    loc: str | Path | ch5mpy.File | ch5mpy.Group,
    dtype: npt.DTypeLike = np.float64,
    chunks: bool | tuple[int, ...] = True,
    maxshape: int | tuple[int | None, ...] | None = None,
) -> ch5mpy.H5Array[Any]:
    return ones.acf(shape, name, loc, dtype, chunks, maxshape)


@with_partial(ArrayCreationFunc("full"))
def full(
    shape: int | tuple[int, ...],
    fill_value: Any,
    name: str,
    loc: str | Path | ch5mpy.File | ch5mpy.Group,
    dtype: npt.DTypeLike = np.float64,
    chunks: bool | tuple[int, ...] = True,
    maxshape: int | tuple[int | None, ...] | None = None,
) -> ch5mpy.H5Array[Any]:
    full.acf._fill_value = fill_value
    return full.acf(shape, name, loc, dtype, chunks, maxshape)
