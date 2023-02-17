# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import TYPE_CHECKING

from ch5mpy.h5array.functions.implement import implements

if TYPE_CHECKING:
    from ch5mpy import H5Array


# ====================================================
# code
_NAN_PLACEHOLDER = object()


@implements(np.unique)
def unique(ar: H5Array[Any],
           return_index: bool = False,
           return_inverse: bool = False,
           return_counts: bool = False,
           axis: int | None = None,
           *,
           equal_nan: bool = True) \
        -> npt.NDArray[Any] | \
        tuple[npt.NDArray[Any], npt.NDArray[np.int_]] | \
        tuple[npt.NDArray[Any], npt.NDArray[np.int_], npt.NDArray[np.int_]] | \
        tuple[npt.NDArray[Any], npt.NDArray[np.int_], npt.NDArray[np.int_], npt.NDArray[np.int_]]:
    if return_inverse:
        raise NotImplementedError

    if axis is not None:
        raise NotImplementedError

    unique = np.array([])
    index = np.array([])
    counts: dict[Any, int] = {}
    index_offset = 0

    for _, chunk in ar.iter_chunks():
        unique_chunk, index_chunk, counts_chunk = \
            np.unique(chunk, return_index=True, return_counts=True, equal_nan=equal_nan)

        unique_concat = np.concatenate((unique, unique_chunk))
        index_concat = np.concatenate((index, index_chunk + index_offset))

        for u, c in zip(unique_chunk, counts_chunk):
            if isinstance(u, float) and np.isnan(u):
                u = _NAN_PLACEHOLDER
            counts[u] = counts.setdefault(u, 0) + c

        unique, i = np.unique(unique_concat, return_index=True, equal_nan=equal_nan)
        index = index_concat[i]

        index_offset += chunk.size

    to_return: tuple[npt.NDArray[Any], ...] = (unique,)
    if return_index:
        to_return += (index,)
    if return_counts:
        counts_array = np.array([counts[u] for u in unique if not np.isnan(u)])
        if _NAN_PLACEHOLDER in counts:
            if equal_nan:
                counts_array = np.concatenate((counts_array, np.array([counts[_NAN_PLACEHOLDER]])))

            else:
                counts_array = np.concatenate((counts_array, np.array([1 for _ in range(counts[_NAN_PLACEHOLDER])])))

        to_return += (counts_array,)

    if len(to_return) == 1:
        return to_return[0]
    return to_return                                                                        # type: ignore[return-value]
