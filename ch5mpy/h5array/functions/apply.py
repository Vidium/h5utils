# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from numpy import _NoValue as NoValue                                                       # type: ignore[attr-defined]
from numbers import Number
from functools import partial

import numpy.typing as npt
from typing import Any
from typing import Union
from typing import Iterable
from typing import TYPE_CHECKING

import ch5mpy
from ch5mpy.h5array.inplace import iter_chunks_2
from ch5mpy.h5array.slice import map_slice
from ch5mpy.h5array.slice import FullSlice

if TYPE_CHECKING:
    from ch5mpy import H5Array


# ====================================================
# code
WHERE_SELECTION = Union[npt.NDArray[np.bool_], tuple[()]]


class Where:
    def __init__(self,
                 where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue,
                 shape: tuple[int, ...]):
        # TODO : make memory efficient (avoid broadcast and compute on the fly)
        self._where = None if where in (True, NoValue) else np.broadcast_to(where, shape)       # type: ignore[arg-type]

    def __getitem__(self, item: tuple[Any, ...] | slice) -> WHERE_SELECTION:
        if self._where is None:
            return ()

        return self._where[item]


def _get_output_array(out: H5Array[Any] | npt.NDArray[Any] | None,
                      shape: tuple[int, ...],
                      axis: tuple[int, ...],
                      keepdims: bool,
                      dtype: npt.DTypeLike | None,
                      initial: int | float | complex | None,
                      default: Any) -> H5Array[Any] | npt.NDArray[Any]:
    if keepdims:
        expected_shape = tuple(s if i not in axis else 1 for i, s in enumerate(shape))

    else:
        expected_shape = tuple(s for i, s in enumerate(shape) if i not in axis)

    if out is None:
        out = np.empty(shape=expected_shape, dtype=dtype)

        if default is not None:
            out[()] = default

    else:
        ndim = len(expected_shape)

        if out.ndim != ndim:
            raise ValueError(f'Output array has the wrong number of dimensions: Found {out.ndim} but expected {ndim}')

        if out.shape != expected_shape:
            raise ValueError(f'Output array has the wrong shape: Found {out.shape} but expected {expected_shape}')

    if initial is not None:
        out[()] = initial

    return out


def _as_tuple(axis: int | Iterable[int] | tuple[int, ...] | None,
              ndim: int,
              default_0D_output: bool) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim)) if default_0D_output else ()

    elif not isinstance(axis, Iterable):
        return axis,

    return tuple(axis)


def _get_indices(index: tuple[FullSlice, ...],
                 axis: tuple[int, ...],
                 where_compute: Where,
                 where_output: Where,
                 output_ndim: int) -> tuple[WHERE_SELECTION, tuple[slice, ...] | tuple[()], WHERE_SELECTION]:
    # compute on whole array at once
    if len(index) == 1 and index[0].is_whole_axis():
        return where_compute[:], (), where_output[index]

    where_to_compute = where_compute[map_slice(index)]

    # 0D output array (no chunk selection, no where selection)
    if output_ndim == 0:
        return where_to_compute, (), ()

    # nD output array
    selected_index = map_slice(tuple(e for i, e in enumerate(index) if i not in axis))
    return where_to_compute, selected_index, where_output[selected_index]


def _apply_operation(operation: str,
                     dest: H5Array[Any] | npt.NDArray[Any],
                     chunk_selection: tuple[slice, ...] | tuple[()],
                     where_to_output: WHERE_SELECTION,
                     values: npt.NDArray[Any]) -> None:
    # Here I have to explicitly write out each operation using +=, *=, ... operators instead of using
    # getattr(..., operation)(...) because dest does not get modified in that way

    if operation == '__set__':
        dest[chunk_selection][where_to_output] = values[where_to_output]

    elif operation == '__iadd__':
        if dest.ndim == 0:
            dest += values                                                                    # type: ignore[assignment]

        else:
            dest[chunk_selection][where_to_output] += values[where_to_output]

    elif operation == '__imul__':
        if dest.ndim == 0:
            dest *= values                                                                    # type: ignore[assignment]

        else:
            dest[chunk_selection][where_to_output] *= values[where_to_output]

    elif operation == '__iand__':
        if dest.ndim == 0:
            dest &= values                                                                    # type: ignore[assignment]

        else:
            dest[chunk_selection][where_to_output] &= values[where_to_output]

    elif operation == '__ior__':
        if dest.ndim == 0:
            dest |= values                                                                    # type: ignore[assignment]

        else:
            dest[chunk_selection][where_to_output] |= values[where_to_output]

    else:
        raise NotImplementedError(f"Do not know how to apply operation '{operation}'")


def apply(func: partial[np.ufunc],
          operation: str,
          a: H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None,
          *,
          dtype: npt.DTypeLike | None,
          initial: int | float | complex | None,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue,
          default_0D_output: bool = True) -> Any:
    axis = _as_tuple(func.keywords.get('axis', None), a.ndim, default_0D_output)
    output_array = _get_output_array(out, a.shape, axis, func.keywords.get('keepdims', False), dtype, initial, None)

    if where is not False:
        where_compute = Where(where, a.shape)
        where_output = Where(where, output_array.shape)

        for index, chunk in a.iter_chunks(keepdims=True):
            where_to_compute, chunk_selection, where_to_output = \
                _get_indices(index, axis, where_compute, where_output, output_array.ndim)
            result = np.array(func(chunk, where=True if where_to_compute == () else where_to_compute),
                              dtype=output_array.dtype)
            _apply_operation(operation, output_array, chunk_selection, where_to_output, result)

    if out is None and output_array.ndim == 0:
        return output_array[()]

    return output_array


def apply_2(func: np.ufunc,
            a: H5Array[Any],
            b: npt.NDArray[Any] | Iterable[Any] | Number | H5Array[Any],
            *,
            out: H5Array[Any] | npt.NDArray[Any] | None,
            default: Any,
            dtype: npt.DTypeLike | None,
            where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue) -> Any:
    output_array = _get_output_array(out, a.shape, (), False, dtype, None, default)

    if where is not False:
        where_compute = Where(where, a.shape)
        where_output = Where(where, output_array.shape)

        if not isinstance(b, (np.ndarray, ch5mpy.H5Array)):
            b = np.array(b)

        for index, chunk_x1, chunk_x2 in iter_chunks_2(a, b):
            where_to_compute, chunk_selection, where_to_output = \
                _get_indices(index, (), where_compute, where_output, output_array.ndim)
            result = np.array(func(chunk_x1, chunk_x2, where=True if where_to_compute == () else where_to_compute),
                              dtype=output_array.dtype)
            _apply_operation('__set__', output_array, chunk_selection, where_to_output, result)

    if out is None and output_array.ndim == 0:
        return output_array[()]

    return output_array
