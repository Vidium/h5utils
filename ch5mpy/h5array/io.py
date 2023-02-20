# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from itertools import zip_longest

from numpy import typing as npt
from typing import Generator
from typing import Union
from typing import TypeVar

from ch5mpy import Dataset
from ch5mpy._typing import SELECTOR
from ch5mpy.h5array.indexing.list import ListIndex
from ch5mpy.h5array.indexing.selection import Selection
from ch5mpy.h5array.indexing.slice import FullSlice
from ch5mpy.objects.dataset import DatasetWrapper
from ch5mpy.utils import is_sequence

# ====================================================
# code
_DT = TypeVar("_DT", bound=np.generic)


def _get_array_sel(sel: tuple[int | npt.NDArray[np.int_] | slice, ...]) -> Generator[int | slice, None, None]:
    for s in sel:
        if isinstance(s, int) or (isinstance(s, np.ndarray) and len(s) == 1):
            yield 0

        else:
            yield slice(None)


def _index_non_slice(elements: tuple[npt.NDArray[np.int_] | slice, ...]) -> int | None:
    for i, e in enumerate(elements):
        if not isinstance(e, slice):
            return i

    return None


def _selection_iter(sel: tuple[npt.NDArray[np.int_] | slice, ...]) -> Generator[
    tuple[tuple[int | npt.NDArray[np.int_] | slice, ...],
          tuple[int | slice, ...]],
    None,
    None
]:
    cut_index = _index_non_slice(sel)

    if len(sel) == 1 or cut_index is None:
        if isinstance(sel[0], np.ndarray) and len(sel[0]) == 1:
            yield (sel[0][0],), tuple(_get_array_sel(sel))

        else:
            yield sel, tuple(_get_array_sel(sel))

    else:
        pre = sel[:cut_index]
        post = sel[cut_index + 1:]
        cut = sel[cut_index]

        if len(cut) == 1:                                                                       # type: ignore[arg-type]
            for p, arr_idx in _selection_iter(post):
                s = cut.start if isinstance(cut, slice) else cut[0]

                yield (*pre, s, *p), (*_get_array_sel(pre), *arr_idx)

        else:
            for i, s in enumerate(cut):                                                         # type: ignore[arg-type]
                for p, arr_idx in _selection_iter(post):
                    yield (*pre, s, *p), (*_get_array_sel(pre), i, *arr_idx)


def read_from_dataset(dataset: Dataset[_DT] | DatasetWrapper[_DT],
                      selection: Selection,
                      loading_array: npt.NDArray[_DT]) -> None:
    for dataset_idx, array_idx in _selection_iter(selection.get()):
        dataset.read_direct(loading_array, source_sel=dataset_idx, dest_sel=array_idx)


def write_to_dataset(dataset: Dataset[_DT] | DatasetWrapper[_DT],
                     values: npt.NDArray[_DT],
                     selection: Selection) -> None:
    if values.ndim == 0:
        values = values.reshape(1)

    for dataset_idx, array_idx in _selection_iter(selection.get()):
        dataset.write_direct(values, source_sel=array_idx, dest_sel=dataset_idx)


def _check_bounds(max_: int, sel: ListIndex | FullSlice, axis: int) -> None:
    valid = True

    if isinstance(sel, FullSlice):
        if sel.start < -max_ or sel.true_stop > max_:
            valid = False

    else:
        if sel.min < -max_ or sel.max > max_:
            valid = False

    if not valid:
        raise IndexError(
            f"Index {sel} is out of bounds for axis {axis} with size {max_}."
        )


def _expanded_index(index: tuple[SELECTOR, ...],
                    shape: tuple[int, ...]) -> Generator[SELECTOR, None, None]:
    """Generate an index whit added zeros where an axis has length 1 in <shape>."""
    i = 0

    for s in shape:
        if s == 1:
            yield 0

        else:
            yield index[i]
            i += 1
            if i == len(index):
                break


def _gets_whole_dataset(index: SELECTOR | tuple[SELECTOR, ...]) -> bool:
    return (isinstance(index, tuple) and index == ()) or \
        (isinstance(index, slice) and index == slice(None)) or \
        (is_sequence(index) and all((e is True for e in index)))


def parse_selector(shape: tuple[int, ...],
                   index: SELECTOR | tuple[SELECTOR, ...]) -> tuple[Selection | None, np.int_]:
    if _gets_whole_dataset(index):
        return None, np.product(shape)

    selection: tuple[ListIndex | FullSlice, ...] = ()

    if not isinstance(index, tuple):
        index = (index,)

    for axis, (axis_len, axis_index) in \
            enumerate(zip_longest(shape, _expanded_index(index, shape), fillvalue=slice(None))):
        if isinstance(axis_len, slice):
            raise RuntimeError

        if isinstance(axis_index, (slice, range)):
            parsed_axis_index: Union[ListIndex, FullSlice] = \
                FullSlice(axis_index.start, axis_index.stop, axis_index.step, axis_len)

        elif is_sequence(axis_index):
            parsed_axis_index = ListIndex(np.array(axis_index))

        elif isinstance(axis_index, int):
            parsed_axis_index = ListIndex(np.array([axis_index]))

        else:
            raise RuntimeError

        _check_bounds(axis_len, parsed_axis_index, axis)
        selection += (parsed_axis_index,)

    return Selection(selection), np.product([len(s) for s in selection])
