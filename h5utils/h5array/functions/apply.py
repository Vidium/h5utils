# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from numpy import _NoValue as NoValue                                                       # type: ignore[attr-defined]
from functools import partial

import numpy.typing as npt
from typing import Any
from typing import Literal
from typing import Callable
from typing import Iterable
from typing import TYPE_CHECKING

from h5utils.h5array.slice import map_slice
from h5utils.h5array.slice import FullSlice

if TYPE_CHECKING:
    from h5utils import H5Array


# ====================================================
# code
class Where:
    def __init__(self,
                 where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue,
                 shape: tuple[int, ...]):
        # TODO : make memory efficient (avoid broadcast)
        self._where = None if where in (True, NoValue) else np.broadcast_to(where, shape)       # type: ignore[arg-type]

    def __getitem__(self, item: tuple[Any, ...] | slice) -> npt.NDArray[np.bool_] | Literal[True]:
        if self._where is None:
            return True

        return self._where[item]


def _get_output_array(out: H5Array[Any] | npt.NDArray[Any] | None,
                      shape: tuple[int, ...],
                      axis: tuple[int, ...],
                      keepdims: bool,
                      dtype: npt.DTypeLike | None,
                      initial: int | float | complex | None) -> H5Array[Any] | npt.NDArray[Any]:
    if keepdims:
        expected_shape = tuple(s if i not in axis else 1 for i, s in enumerate(shape))

    else:
        expected_shape = tuple(s for i, s in enumerate(shape) if i not in axis)

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


def _as_tuple(axis: int | Iterable[int] | tuple[int, ...] | None,
              ndim: int) -> tuple[int, ...]:
    if axis is None:
        return tuple(range(ndim))

    elif not isinstance(axis, Iterable):
        return axis,

    return tuple(axis)


def _get_indices(index: tuple[FullSlice, ...],
                 axis: tuple[int, ...],
                 where: Where) -> tuple[tuple[slice, ...] | tuple[()], npt.NDArray[np.bool_] | Literal[True]]:
    if len(index) == 1 and index[0].is_whole_axis():
        return (), where[:]

    else:
        return map_slice((e for i, e in enumerate(index) if i not in axis)), where[map_slice(index)]


def apply(func: partial[Callable[..., npt.NDArray[Any] | Iterable[Any] | int | float]],
          operation: str,
          a: H5Array[Any],
          out: H5Array[Any] | npt.NDArray[Any] | None,
          *,
          dtype: npt.DTypeLike | None,
          initial: int | float | complex | None,
          where: npt.NDArray[np.bool_] | Iterable[np.bool_] | int | bool | NoValue) -> Any:
    axis = _as_tuple(func.keywords['axis'], a.ndim)
    output_array = _get_output_array(out, a.shape, axis, func.keywords['keepdims'], dtype, initial)

    if where is not False:
        for index, chunk in a.iter_chunks(keepdims=True):
            out_index, comp_index = _get_indices(index, axis, Where(where, a.shape))

            if output_array.ndim:
                getattr(output_array[out_index], operation)(
                    np.array(func(chunk, where=comp_index), dtype=output_array.dtype)
                )

            else:
                getattr(output_array, operation)(
                    np.array(func(chunk, where=comp_index), dtype=output_array.dtype)
                )

    if out is None:
        return output_array[()]

    return output_array
