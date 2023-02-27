# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import cast

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import Iterable
from typing import TYPE_CHECKING

import ch5mpy
from ch5mpy._typing import NP_FUNC
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


def _in_chunk(chunk_1: npt.NDArray[Any], chunk_2: npt.NDArray[Any], res: npt.NDArray[Any], invert: bool,
              func: NP_FUNC) -> npt.NDArray[np.bool_]:
    if invert:
        return np.logical_and(res, func(chunk_1, chunk_2, invert=True))

    else:
        return np.logical_or(res, func(chunk_1, chunk_2))


@implements(np.in1d)
def in1d(ar1: Any,
         ar2: Any,
         invert: bool = False) -> npt.NDArray[np.bool_]:
    # cast arrays as either np.arrays or H5Arrays
    if not isinstance(ar1, (np.ndarray, ch5mpy.H5Array)):
        ar1 = np.array(ar1)

    if not isinstance(ar2, (np.ndarray, ch5mpy.H5Array)):
        ar2 = np.array(ar2)

    # prepare output
    if invert:
        res = np.ones(ar1.size, dtype=bool)
    else:
        res = np.zeros(ar1.size, dtype=bool)

    # case np.array in H5Array
    if isinstance(ar1, np.ndarray):
        ar2 = cast(ch5mpy.H5Array[Any], ar2)

        for _, chunk in ar2.iter_chunks():
            res = _in_chunk(ar1, chunk, res, invert=invert, func=np.in1d)

    else:
        ar1 = cast(ch5mpy.H5Array[Any], ar1)
        index_offset = 0

        # case H5Array in np.array
        if isinstance(ar2, np.ndarray):
            for _, chunk in ar1.iter_chunks():
                index = slice(index_offset, index_offset + chunk.size)
                index_offset += chunk.size
                res[index] = _in_chunk(chunk, ar2, res[index], invert=invert, func=np.in1d)

        # case H5Array in H5Array
        else:
            ar2 = cast(ch5mpy.H5Array[Any], ar2)

            for _, chunk_1 in ar1.iter_chunks():
                index = slice(index_offset, index_offset + chunk_1.size)
                index_offset += chunk_1.size

                for _, chunk_2 in ar2.iter_chunks():
                    res[index] = _in_chunk(chunk_1, chunk_2, res[index], invert=invert, func=np.in1d)

    return res


@implements(np.isin)
def isin(element: Any,
         test_elements: Any,
         invert: bool = False) -> npt.NDArray[np.bool_]:
    # cast arrays as either np.arrays or H5Arrays
    if not isinstance(element, (np.ndarray, ch5mpy.H5Array)):
        element = np.array(element)

    if not isinstance(test_elements, (np.ndarray, ch5mpy.H5Array)):
        test_elements = np.array(test_elements)

    # prepare output
    if invert:
        res = np.ones_like(element, dtype=bool)
    else:
        res = np.zeros_like(element, dtype=bool)

    # case np.array in H5Array
    if isinstance(element, np.ndarray):
        test_elements = cast(ch5mpy.H5Array[Any], test_elements)

        for _, chunk in test_elements.iter_chunks():
            res = _in_chunk(element, chunk, res, invert=invert, func=np.isin)

    else:
        element = cast(ch5mpy.H5Array[Any], element)

        # case H5Array in np.array
        if isinstance(test_elements, np.ndarray):
            for index, chunk in element.iter_chunks():
                res[index] = _in_chunk(chunk, test_elements, res[index], invert=invert, func=np.isin)

        # case H5Array in H5Array
        else:
            test_elements = cast(ch5mpy.H5Array[Any], test_elements)

            for index_elem, chunk_elem in element.iter_chunks():
                for _, chunk_test in test_elements.iter_chunks():
                    res[index_elem] = _in_chunk(chunk_elem, chunk_test, res[index_elem], invert=invert, func=np.isin)

    return res


@implements(np.concatenate)
def concatenate(arrays: H5Array[Any] | tuple[H5Array[Any] | npt.NDArray[Any], ...],
                axis: int | None = 0,
                out: H5Array[Any] | npt.NDArray[Any] | None = None,
                dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
    if out is None:
        if isinstance(arrays, ch5mpy.H5Array):
            return np.concatenate(np.array(arrays), axis=axis, dtype=dtype)

        else:
            return np.concatenate(tuple(map(np.array, arrays)), axis=axis, dtype=dtype)

    else:
        raise NotImplementedError


@implements(np.hstack)
def hstack(tup: Iterable[H5Array[Any] | npt.NDArray[Any]],
           *,
           dtype: npt.DTypeLike | None = None,
           casting: str = "same_kind") -> npt.NDArray[Any]:
    return np.hstack(map(np.array, tup), dtype=dtype, casting=casting)      # type: ignore[no-any-return, call-overload]


@implements(np.vstack)
def vstack(tup: Iterable[H5Array[Any] | npt.NDArray[Any]],
           *,
           dtype: npt.DTypeLike | None = None,
           casting: str = "same_kind") -> npt.NDArray[Any]:
    return np.vstack(map(np.array, tup), dtype=dtype, casting=casting)      # type: ignore[no-any-return, call-overload]
