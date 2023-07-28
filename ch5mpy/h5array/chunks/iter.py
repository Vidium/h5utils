from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Any, Generator, TypeVar, cast

import numpy as np
import numpy.typing as npt

import ch5mpy
from ch5mpy.h5array.chunks.repeated_array import RepeatedArray
from ch5mpy.h5array.chunks.utils import _as_valid_dtype
from ch5mpy.indexing import FullSlice, SingleIndex, map_slice

if TYPE_CHECKING:
    from ch5mpy import H5Array


_DT = TypeVar("_DT", bound=np.generic)
INF = np.iinfo(int).max


def get_work_array(
    shape: tuple[int, ...], slicer: tuple[FullSlice | SingleIndex, ...], dtype: np.dtype[_DT]
) -> npt.NDArray[_DT]:
    if len(slicer) == 1 and isinstance(slicer[0], FullSlice) and slicer[0].is_whole_axis:
        return np.empty(shape, dtype=object if np.issubdtype(dtype, str) else dtype)

    slicer_shape = tuple(len(s) for s in slicer if not isinstance(s, SingleIndex))
    return np.empty(slicer_shape, dtype=object if np.issubdtype(dtype, str) else dtype)


def _get_chunk_indices(chunk_size: int, shape: tuple[int, ...]) -> tuple[tuple[FullSlice | SingleIndex, ...], ...]:
    # special case of 0D arrays
    if len(shape) == 0:
        raise ValueError("0D array")

    if np.prod(shape) == 0:
        return (tuple(FullSlice.whole_axis(s) for s in shape),)

    rev_shape = tuple(reversed(shape))

    # not enough max mem
    if chunk_size <= 1:
        raise ValueError("Slicing is impossible because of insufficient allowed memory usage.")

    block_axes = int(np.argmax(~(np.cumprod(rev_shape + (np.inf,)) <= chunk_size)))
    size_block = chunk_size // np.cumprod(rev_shape)[block_axes - 1] if block_axes else min(rev_shape[0], chunk_size)

    if size_block == 0:
        block_axes = max(0, block_axes - 1)

    if block_axes == len(shape):
        # all array can be read at once
        return (tuple(FullSlice.whole_axis(s) for s in shape),)

    whole_axes = tuple(FullSlice.whole_axis(s) for s in rev_shape[:block_axes][::-1])
    iter_axis = rev_shape[block_axes]

    right_chunks = tuple(
        (FullSlice(s, min(s + size_block, iter_axis), 1, iter_axis), *whole_axes)
        for s in range(0, iter_axis, size_block)
    )

    if block_axes + 1 == len(shape):
        return right_chunks

    left_shape = shape[: -(block_axes + 1)]
    left_chunks = np.array(np.meshgrid(*map(range, left_shape))).T.reshape(-1, len(left_shape))

    return tuple(
        tuple(SingleIndex(_l, _s) for _l, _s in zip(left, shape)) + tuple(right)
        for left in left_chunks
        for right in right_chunks
    )


class ChunkIterator:
    def __init__(self, array: H5Array[Any], keepdims: bool = False):
        self._array = array
        self._keepdims = keepdims

        self._chunk_indices = _get_chunk_indices(array.chunk_size, array.shape)

        self._work_array = get_work_array(array.shape, self._chunk_indices[0], dtype=array.dtype)

    def __repr__(self) -> str:
        return f"<ChunkIterator over {self._array.shape} H5Array>"

    def __iter__(
        self,
    ) -> Generator[tuple[tuple[FullSlice | SingleIndex, ...], npt.NDArray[Any]], None, None]:
        for index in self._chunk_indices:
            work_subset = map_slice(index, shift_to_zero=True)
            self._array.read_direct(self._work_array, source_sel=map_slice(index), dest_sel=work_subset)

            # cast to str if needed
            res = _as_valid_dtype(self._work_array, self._array.dtype)[work_subset]

            # reshape to keep dimensions if needed
            if self._keepdims:
                res = res.reshape((1,) * (self._array.ndim - res.ndim) + res.shape)

            yield index, res


class PairedChunkIterator:
    def __init__(
        self,
        arr_1: H5Array[Any] | npt.NDArray[Any],
        arr_2: H5Array[Any] | npt.NDArray[Any],
        keepdims: bool = False,
    ):
        broadcasted_shape = np.broadcast_shapes(arr_1.shape, arr_2.shape)
        self._arr_1 = RepeatedArray(arr_1, broadcasted_shape)
        self._arr_2 = RepeatedArray(arr_2, broadcasted_shape)

        self._keepdims = keepdims

        chunk_size = min(
            arr_1.chunk_size if isinstance(arr_1, ch5mpy.H5Array) else INF,
            arr_2.chunk_size if isinstance(arr_2, ch5mpy.H5Array) else INF,
        )

        self._chunk_indices = _get_chunk_indices(chunk_size, shape=arr_1.shape)

        self._work_array_1 = get_work_array(broadcasted_shape, self._chunk_indices[0], dtype=arr_1.dtype)
        self._work_array_2 = get_work_array(broadcasted_shape, self._chunk_indices[0], dtype=arr_2.dtype)

    def __repr__(self) -> str:
        return f"<PairedChunkIterator over 2 {self._arr_1.shape} arrays>"

    def __iter__(
        self,
    ) -> Generator[tuple[tuple[SingleIndex | FullSlice, ...], npt.NDArray[Any], npt.NDArray[Any]], None, None]:
        for index in self._chunk_indices:
            work_subset = map_slice(index, shift_to_zero=True)
            res_1 = self._arr_1.read(self._work_array_1, map_slice(index), work_subset)
            res_2 = self._arr_2.read(self._work_array_2, map_slice(index), work_subset)

            if self._keepdims:
                res_1 = res_1.reshape((1,) * (self._arr_1.ndim - res_1.ndim) + res_1.shape)
                res_2 = res_2.reshape((1,) * (self._arr_2.ndim - res_2.ndim) + res_2.shape)

            yield index, res_1, res_2


def iter_chunks_2(
    x1: npt.NDArray[Any] | H5Array[Any], x2: npt.NDArray[Any] | H5Array[Any]
) -> Generator[tuple[tuple[SingleIndex | FullSlice, ...], npt.NDArray[Any], npt.NDArray[Any] | Number], None, None,]:
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
