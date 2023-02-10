# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import TypeVar


# ====================================================
# code
_DT = TypeVar("_DT", bound=np.generic)


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


def get_chunks(
    max_memory_usage: int | str, shape: tuple[int, ...], itemsize: int
) -> tuple[tuple[int | slice, ...], ...]:
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
        return ((slice(None),),)

    if size_block == 1:
        right_chunks = tuple(
            (s,) + (slice(None),) * block_axes for s in range(0, rev_shape[block_axes])
        )

    else:
        right_chunks = tuple(
            (slice(s, min(s + size_block, rev_shape[block_axes])),)
            + (slice(None),) * block_axes
            for s in range(0, rev_shape[block_axes], size_block)
        )

    if block_axes + 1 == len(shape):
        return right_chunks

    left_shape = shape[:-(block_axes+1)]
    left_chunks = np.array(np.meshgrid(*map(range, left_shape))).T.reshape(
        -1, len(left_shape)
    )

    return tuple(
        tuple(left) + tuple(right) for left in left_chunks for right in right_chunks
    )


def _len(obj: int | slice) -> int:
    if isinstance(obj, slice):
        return int(obj.stop - obj.start)

    return 1


def get_work_array(shape: tuple[int, ...], slicer: tuple[int | slice, ...], dtype: _DT) -> npt.NDArray[_DT]:
    if slicer == (slice(None),):
        return np.empty(shape, dtype=dtype)

    slicer_shape = tuple(shape[i] if s == slice(None) else _len(s)
                         for i, s in enumerate(slicer))
    return np.empty(slicer_shape, dtype=dtype)


def get_work_sel(slicer: tuple[int | slice, ...]) -> tuple[int | slice, ...]:
    sel: tuple[int | slice, ...] = ()
    for s in slicer:
        if s == slice(None):
            sel += (slice(None),)

        elif isinstance(s, slice):
            sel += (slice(0, s.stop - s.start),)

        else:
            sel += (0,)

    return sel
