# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from numpy import _NoValue as NoValue                                                       # type: ignore[attr-defined]

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import TypeVar
from typing import Callable
from typing import Iterable
from typing import TYPE_CHECKING

import h5utils
from h5utils._typing import NP_FUNC
from h5utils._typing import H5_FUNC
from h5utils.h5array.inplace import get_chunks
from h5utils.h5array.inplace import get_work_array
from h5utils.h5array.inplace import get_work_sel
from h5utils.utils import is_sequence

if TYPE_CHECKING:
    from h5utils import H5Array
    from h5utils import Dataset

# ====================================================
# code
_T = TypeVar("_T", bound=np.generic, covariant=True)

_HANDLED_FUNCTIONS: dict[NP_FUNC | np.ufunc, H5_FUNC] = {}


def implements(np_function: NP_FUNC | np.ufunc) -> Callable[[H5_FUNC], H5_FUNC]:
    """Register an __array_function__ implementation for DiagonalArray objects."""

    def decorator(func: H5_FUNC) -> H5_FUNC:
        _HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


def _get_output_array(out: H5Array[Any] | npt.NDArray[Any] | None,
                      shape: tuple[int, ...],
                      axis: int | Iterable[int] | tuple[int] | None,
                      keepdims: bool,
                      dtype: npt.DTypeLike | None,
                      initial: int | float | complex | None) -> H5Array[Any] | npt.NDArray[Any]:
    if axis is None:
        expected_shape: tuple[int, ...] = ()

    else:
        if isinstance(axis, int):
            axis = (axis,)

        expected_shape = tuple(s for i, s in enumerate(shape) if i not in axis)

    if keepdims:
        expected_shape = (1,) * (len(shape) - len(expected_shape)) + expected_shape

    if out is not None:
        ndim = len(expected_shape)

        if out.ndim != ndim:
            raise ValueError(f'Output array has the wrong number of dimensions: Found {out.ndim} but expected {ndim}')

        if out.shape != expected_shape:
            raise ValueError(f'Output array has the wrong shape: Found {out.shape} but expected {expected_shape}')

    else:
        out = np.empty(shape=expected_shape, dtype=dtype)

    out[()] = 0 if initial is None else initial
    return out


def _nest(obj: int, degree: int) -> tuple[Any]:
    obj_: tuple[Any] = (obj,)

    for _ in range(degree-1):
        obj_ = (obj_,)

    return obj_


def _cast_chunk_index(index: tuple[int | slice, ...],
                      axis: int | Iterable[int] | tuple[int] | None) -> tuple[tuple[Any] | int | slice, ...]:
    if index == (slice(None),):
        return ()

    if axis is None:
        return index

    if not isinstance(axis, Iterable):
        axis = (axis,)

    return tuple(e if isinstance(e, slice) else _nest(e, len(index) - len(tuple(axis)))
                 for i, e in enumerate(index) if i not in axis)


@implements(np.sum)
def sum(a: H5Array[Any],
        axis: int | Iterable[int] | tuple[int] | None = None,
        dtype: npt.DTypeLike | None = None,
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        keepdims: bool = False,
        initial: int | float | complex | None = None,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue = NoValue) -> Any:
    accumulator = _get_output_array(out, a.shape, axis, keepdims, dtype, initial)

    for index, chunk in a.iter_chunks(keepdims=True):
        acc_index = _cast_chunk_index(index, axis)
        accumulator[acc_index] += np.array(
            np.sum(chunk, keepdims=keepdims, dtype=dtype, axis=axis, where=where),    # type: ignore[arg-type, operator]
            dtype=accumulator.dtype
        )

    if out is None and accumulator.ndim == 0:
        return accumulator[()]

    return accumulator


@implements(np.mean)
def mean(a: H5Array[Any],
         axis: int | Iterable[Any] | tuple[int] | None = None,
         out: H5Array[Any] | npt.NDArray[Any] | None = None,
         keepdims: bool | None = None,
         *,
         where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | None = None,
         **kwargs: Any) -> Any:
    return np.mean(a.dset, axis=axis)  # type: ignore[call-overload]


@implements(np.all)
def all(a: H5Array[Any],
        axis: int | Iterable[Any] | tuple[int] | None = None,
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        keepdims: bool | None = None,
        *,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | None = None,
        **kwargs: Any) -> npt.NDArray[Any] | bool:
    if axis is not None or keepdims is not None or where is not None:
        raise NotImplementedError

    if out is not None and (not isinstance(out, npt.NDArray) or out.ndim != 0):
        raise ValueError('Expected one NDArray with 0 dimensions.')

    chunks = get_chunks(a.MAX_MEM_USAGE, a.shape, a.dtype.itemsize)
    work_array = get_work_array(a.shape, chunks[0], dtype=a.dtype)
    res = out or True

    for chunk in chunks:
        work_subset = get_work_sel(chunk)
        a.dset.read_direct(work_array, source_sel=chunk, dest_sel=work_subset)

        if not np.all(work_array[work_subset]):
            if out is None:
                res = False
            else:
                cast(npt.NDArray[Any], res)[()] = False

            break

    return res


@implements(np.any)
def any(a: H5Array[Any],
        axis: int | Iterable[Any] | tuple[int] | None = None,
        out: H5Array[Any] | npt.NDArray[Any] | None = None,
        keepdims: bool | None = None,
        *,
        where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | None = None,
        **kwargs: Any) -> npt.NDArray[Any] | bool:
    if axis is not None or keepdims is not None or where is not None:
        raise NotImplementedError

    if out is not None and (not isinstance(out, npt.NDArray) or out.ndim != 0):
        raise ValueError('Expected one NDArray with 0 dimensions.')

    chunks = get_chunks(a.MAX_MEM_USAGE, a.shape, a.dtype.itemsize)
    work_array = get_work_array(a.shape, chunks[0], dtype=a.dtype)
    res = out or False

    for chunk in chunks:
        work_subset = get_work_sel(chunk)
        a.dset.read_direct(work_array, source_sel=chunk, dest_sel=work_subset)

        if np.any(work_array[work_subset]):
            if out is None:
                res = True
            else:
                cast(npt.NDArray[Any], res)[()] = True

            break

    return res


def _set_false(r: H5Array[Any] | npt.NDArray[Any] | bool) -> H5Array[Any] | npt.NDArray[Any] | bool:
    if isinstance(r, bool):
        return False

    r[()] = False
    return r


def _cast_H5Array(obj: Any) -> H5Array[Any]:
    return obj                                                                             # type: ignore[no-any-return]


@implements(np.equal)
def equal(x1: int | float | complex | str | bytes | bool | np.generic | npt.NDArray[Any] | H5Array[Any],
          x2: int | float | complex | str | bytes | bool | np.generic | npt.NDArray[Any] | H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None = None,
          *,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | None = None,
          **kwargs: Any) -> H5Array[Any] | npt.NDArray[Any] | bool:
    if where is not None:
        raise NotImplementedError

    if out is not None and (not isinstance(out, npt.NDArray) or out.ndim != 0):
        raise ValueError('Expected one NDArray with 0 dimensions.')

    res: H5Array[Any] | npt.NDArray[Any] | bool = out or True

    if not isinstance(x1, h5utils.H5Array):
        # ensure x1 is an H5Array
        x1, x2 = _cast_H5Array(x2), x1

    if isinstance(x2, h5utils.H5Array):
        if x1.shape != x2.shape:
            return _set_false(res)

        chunks = get_chunks(min(x1.MAX_MEM_USAGE, x2.MAX_MEM_USAGE),
                            x1.shape,
                            max(x1.dtype.itemsize, x2.dtype.itemsize))

        work_array_x1 = get_work_array(x1.shape, chunks[0], dtype=x1.dtype)
        work_array_x2 = get_work_array(x1.shape, chunks[0], dtype=x2.dtype)

        for chunk in chunks:
            work_subset = get_work_sel(chunk)
            x1.dset.read_direct(work_array_x1, source_sel=chunk, dest_sel=work_subset)
            x2.dset.read_direct(work_array_x2, source_sel=chunk, dest_sel=work_subset)

            if not np.equal(work_array_x1[work_subset], work_array_x2[work_subset]):
                res = _set_false(res)
                break

        return res

    elif is_sequence(x2):
        x2 = np.array(x2)

        chunks = get_chunks(x1.MAX_MEM_USAGE, x1.shape, x1.dtype.itemsize)
        work_array = get_work_array(x1.shape, chunks[0], dtype=x1.dtype)

        for chunk in chunks:
            work_subset = get_work_sel(chunk)
            x1.dset.read_direct(work_array, source_sel=chunk, dest_sel=work_subset)

            if not np.equal(work_array[work_subset], x2[work_subset]):
                res = _set_false(res)
                break

        return res

    else:
        chunks = get_chunks(x1.MAX_MEM_USAGE, x1.shape, x1.dtype.itemsize)
        work_array = get_work_array(x1.shape, chunks[0], dtype=x1.dtype)

        for chunk in chunks:
            work_subset = get_work_sel(chunk)
            x1.dset.read_direct(work_array, source_sel=chunk, dest_sel=work_subset)

            if not np.equal(work_array[work_subset], x2):
                res = _set_false(res)
                break

        return res


@implements(np.array_equal)
def array_equal(x1: npt.NDArray[Any] | Iterable[Any] | int | float | H5Array[Any],
                x2: npt.NDArray[Any] | Iterable[Any] | int | float | H5Array[Any],
                equal_nan: bool = False) -> bool:
    if not isinstance(x1, h5utils.H5Array):
        # ensure x1 is an H5Array
        x1, x2 = _cast_H5Array(x2), x1

    if isinstance(x2, h5utils.H5Array):
        if x1.shape != x2.shape:
            return False






    if isinstance(x1, h5utils.H5Array):
        arr1_data: npt.NDArray[Any] | Iterable[Any] | int | float | Dataset[Any] = np.array(x1)
    else:
        arr1_data = x1

    if isinstance(x2, h5utils.H5Array):
        arr2_data: npt.NDArray[Any] | Iterable[Any] | int | float | Dataset[Any] = np.array(x2)
    else:
        arr2_data = x2

    return np.array_equal(arr1_data, arr2_data, equal_nan=equal_nan)                            # type: ignore[arg-type]
