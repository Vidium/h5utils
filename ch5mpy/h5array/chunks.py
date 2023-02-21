# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from numbers import Number

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import TypeVar
from typing import Generator
from typing import TYPE_CHECKING

import ch5mpy
from ch5mpy.h5array.indexing.slice import FullSlice
from ch5mpy.h5array.indexing.slice import map_slice

if TYPE_CHECKING:
    from ch5mpy import H5Array


# ====================================================
# code
_DT = TypeVar("_DT", bound=np.generic)
INF = np.iinfo(int).max
SIZES = {"K": 1024,
         "M": 1024 * 1024,
         "G": 1024 * 1024 * 1024}


def get_size(s: int | str) -> int:
    value: int | None = None

    if isinstance(s, int):
        value = s

    elif s[-1] in SIZES and s[:-1].lstrip("-").isdigit():
        value = int(s[:-1]) * SIZES[s[-1]]

    elif s.isdigit():
        value = int(s)

    if value is None:
        raise ValueError(f"Unrecognized size '{s}'")

    if value <= 0:
        raise ValueError(f"Got invalid size ({value} <= 0).")

    return value


def get_work_array(shape: tuple[int, ...],
                   slicer: tuple[FullSlice, ...],
                   dtype: np.dtype[_DT]) -> npt.NDArray[_DT]:
    if len(slicer) == 1 and slicer[0].is_whole_axis():
        return np.empty(shape, dtype=object if np.issubdtype(dtype, str) else dtype)

    slicer_shape = tuple(len(s) for s in slicer)
    return np.empty(slicer_shape, dtype=object if np.issubdtype(dtype, str) else dtype)


def _get_chunk_indices(max_memory_usage: int | str,
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
        return tuple(FullSlice.whole_axis(s) for s in shape),

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


def _valid_dtype(arr: npt.NDArray[Any], dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    if np.issubdtype(dtype, str):
        return arr.astype(str)

    return arr


class ChunkIterator:
    def __init__(self,
                 array: H5Array[Any],
                 keepdims: bool = False):
        self._array = array
        self._keepdims = keepdims

        self._chunk_indices = _get_chunk_indices(array.MAX_MEM_USAGE, array.shape, array.dtype.itemsize)
        self._work_array = get_work_array(array.shape, self._chunk_indices[0], dtype=array.dtype)

    def __repr__(self) -> str:
        return f"<ChunkIterator over {self._array.shape} H5Array>"

    def __iter__(self) -> Generator[tuple[tuple[FullSlice, ...], npt.NDArray[Any]], None, None]:
        for index in self._chunk_indices:
            work_subset = map_slice(c.shift_to_zero() for c in index)
            self._array.read_direct(self._work_array, source_sel=map_slice(index), dest_sel=work_subset)

            # cast to str if needed
            res = _valid_dtype(self._work_array, self._array.dtype)[work_subset]

            # reshape to keep dimensions if needed
            if self._keepdims:
                res = res.reshape((1,) * (self._array.ndim - res.ndim) + res.shape)

            yield index, res


class PairedChunkIterator:
    def __init__(self,
                 arr_1: H5Array[Any] | npt.NDArray[Any],
                 arr_2: H5Array[Any] | npt.NDArray[Any],
                 keepdims: bool = False):
        if arr_1.shape != arr_2.shape:
            raise ValueError(f'Cannot iterate chunks of arrays with different shapes: {arr_1.shape} != {arr_2.shape}')

        self._arr_1 = arr_1
        self._arr_2 = arr_2
        self._keepdims = keepdims

        max_mem_1 = get_size(arr_1.MAX_MEM_USAGE) if isinstance(arr_1, ch5mpy.H5Array) else INF
        max_mem_2 = get_size(arr_2.MAX_MEM_USAGE) if isinstance(arr_2, ch5mpy.H5Array) else INF

        self._chunk_indices = _get_chunk_indices(min(max_mem_1, max_mem_2),
                                                 shape=arr_1.shape,
                                                 itemsize=max(arr_1.dtype.itemsize, arr_1.dtype.itemsize))
        self._work_array_1 = get_work_array(arr_1.shape, self._chunk_indices[0], dtype=arr_1.dtype)
        self._work_array_2 = get_work_array(arr_1.shape, self._chunk_indices[0], dtype=arr_2.dtype)

    def __repr__(self) -> str:
        return f"<PairedChunkIterator over 2 {self._arr_1.shape} H5Arrays>"

    @staticmethod
    def _read_array(arr: npt.NDArray[Any] | H5Array[Any],
                    out: npt.NDArray[Any],
                    source_sel: tuple[slice, ...],
                    dest_sel: tuple[slice, ...]) -> None:
        if isinstance(arr, ch5mpy.H5Array):
            arr.read_direct(out, source_sel=source_sel, dest_sel=dest_sel)

        else:
            out[dest_sel] = arr[source_sel]

    def __iter__(self) -> Generator[tuple[tuple[FullSlice, ...], npt.NDArray[Any], npt.NDArray[Any]], None, None]:
        for index in self._chunk_indices:
            work_subset = map_slice(c.shift_to_zero() for c in index)
            self._read_array(self._arr_1, self._work_array_1, map_slice(index), work_subset)
            self._read_array(self._arr_2, self._work_array_2, map_slice(index), work_subset)

            res_1 = _valid_dtype(self._work_array_1, self._arr_1.dtype)[work_subset]
            res_2 = _valid_dtype(self._work_array_2, self._arr_2.dtype)[work_subset]

            if self._keepdims:
                res_1 = res_1.reshape((1,) * (self._arr_1.ndim - res_1.ndim) + res_1.shape)
                res_2 = res_2.reshape((1,) * (self._arr_2.ndim - res_2.ndim) + res_2.shape)

            yield index, res_1, res_2


def iter_chunks_2(x1: npt.NDArray[Any] | H5Array[Any],
                  x2: npt.NDArray[Any] | H5Array[Any]) \
        -> Generator[tuple[tuple[FullSlice, ...], npt.NDArray[Any], npt.NDArray[Any] | Number], None, None]:
    # special case where x2 is a 0D array, iterate through chunks of x1 and always yield x2
    if x2.ndim == 0:
        if isinstance(x1, ch5mpy.H5Array):
            for chunk, arr in ChunkIterator(x1):
                yield chunk, arr, cast(Number, x2[()])

        else:
            yield (FullSlice.whole_axis(x1.shape[0]),), x1, cast(Number, x2[()])

    # nD case
    else:
        yield from PairedChunkIterator(x1, x2)
