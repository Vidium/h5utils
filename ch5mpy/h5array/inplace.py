# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from numbers import Number
from typing import Generator
from typing import cast

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import TypeVar
from typing import TYPE_CHECKING

import ch5mpy
from ch5mpy.h5array.slice import FullSlice

if TYPE_CHECKING:
    from ch5mpy import H5Array

# ====================================================
# code
_DT = TypeVar("_DT", bound=np.generic)
INF = np.iinfo(int).max


sizes = {"K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}


def get_size(s: int | str) -> int:
    value: int | None = None

    if isinstance(s, int):
        value = s

    elif s[-1] in sizes and s[:-1].lstrip("-").isdigit():
        value = int(s[:-1]) * sizes[s[-1]]

    elif s.isdigit():
        value = int(s)

    if value is None:
        raise ValueError(f"Unrecognized size '{s}'")

    if value <= 0:
        raise ValueError(f"Got invalid size ({value} <= 0).")

    return value


def get_chunks(max_memory_usage: int | str,
               shape: tuple[int, ...],
               itemsize: int) -> tuple[tuple[FullSlice, ...], ...]:
    # special case of 0D arrays
    if len(shape) == 0:
        raise ValueError("0D array")

    rev_shape = tuple(reversed(shape))
    nb_elements_chunk = int(get_size(max_memory_usage) / itemsize)

    # not enough max mem
    if nb_elements_chunk <= 1:
        raise ValueError(
            "Slicing is impossible because of insufficient allowed memory usage."
        )

    block_axes = int(np.argmax(~(np.cumprod(rev_shape + (np.inf,)) <= nb_elements_chunk)))
    size_block = (
        nb_elements_chunk // np.cumprod(rev_shape)[block_axes - 1]
        if block_axes
        else min(rev_shape[0], nb_elements_chunk)
    )

    if size_block == 0:
        block_axes = max(0, block_axes - 1)

    if block_axes == len(shape):
        # all array can be read at once
        return ((FullSlice.whole_axis(shape[0]),),)

    whole_axes = tuple(FullSlice.whole_axis(s) for s in rev_shape[:block_axes][::-1])
    iter_axis = rev_shape[block_axes]

    right_chunks = tuple(
        (FullSlice(s, min(s + size_block, iter_axis), 1, iter_axis), *whole_axes)
        for s in range(0, iter_axis, size_block)
    )

    if block_axes + 1 == len(shape):
        return right_chunks

    left_shape = shape[:-(block_axes+1)]
    left_chunks = np.array(np.meshgrid(*map(range, left_shape))).T.reshape(
        -1, len(left_shape)
    )

    return tuple(
        tuple(map(FullSlice.one, left, shape)) + tuple(right) for left in left_chunks for right in right_chunks
    )


def _len(obj: int | FullSlice) -> int:
    if isinstance(obj, FullSlice):
        return len(obj)

    return 1


def get_work_array(shape: tuple[int, ...],
                   slicer: tuple[FullSlice, ...],
                   dtype: np.dtype[_DT]) -> npt.NDArray[_DT]:
    if len(slicer) == 1 and slicer[0].is_whole_axis():
        return np.empty(shape, dtype=dtype)

    slicer_shape = tuple(_len(s) for s in slicer)
    return np.empty(slicer_shape, dtype=dtype)


def _read_array(arr: npt.NDArray[Any] | H5Array[Any],
                out: npt.NDArray[Any],
                source_sel: tuple[FullSlice, ...],
                dest_sel: tuple[FullSlice, ...]) -> None:
    if isinstance(arr, ch5mpy.H5Array):
        arr.read_direct(out, source_sel=source_sel, dest_sel=dest_sel)

    else:
        out[dest_sel] = arr[source_sel]


def iter_chunks_2(x1: npt.NDArray[Any] | H5Array[Any],
                  x2: npt.NDArray[Any] | H5Array[Any]) \
        -> Generator[tuple[tuple[FullSlice, ...], npt.NDArray[Any], npt.NDArray[Any] | Number], None, None]:
    # special case where x2 is a 0D array, iterate through chunks of x1 and always yield x2
    if x2.ndim == 0:
        max_mem_x1 = get_size(x1.MAX_MEM_USAGE) if isinstance(x1, ch5mpy.H5Array) else INF

        chunks = get_chunks(max_mem_x1, x1.shape, x1.dtype.itemsize)
        work_array_x1 = get_work_array(x1.shape, chunks[0], dtype=x1.dtype)

        for chunk in chunks:
            work_subset = tuple(c.shift_to_zero() for c in chunk)
            _read_array(x1, work_array_x1, chunk, work_subset)

            yield chunk, work_array_x1[work_subset], cast(Number, x2[()])

    # nD case
    else:
        if x1.shape != x2.shape:
            raise ValueError(f'Cannot iterate chunks of arrays with different shapes: {x1.shape} != {x2.shape}')

        max_mem_x1 = get_size(x1.MAX_MEM_USAGE) if isinstance(x1, ch5mpy.H5Array) else INF
        max_mem_x2 = get_size(x2.MAX_MEM_USAGE) if isinstance(x2, ch5mpy.H5Array) else INF

        chunks = get_chunks(min(max_mem_x1, max_mem_x2), x1.shape, max(x1.dtype.itemsize, x2.dtype.itemsize))
        work_array_x1 = get_work_array(x1.shape, chunks[0], dtype=x1.dtype)
        work_array_x2 = get_work_array(x1.shape, chunks[0], dtype=x2.dtype)

        for chunk in chunks:
            work_subset = tuple(c.shift_to_zero() for c in chunk)
            _read_array(x1, work_array_x1, chunk, work_subset)
            _read_array(x2, work_array_x2, chunk, work_subset)

            yield chunk, work_array_x1[work_subset], work_array_x2[work_subset]
