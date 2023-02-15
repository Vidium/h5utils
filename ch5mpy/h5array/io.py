# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from itertools import zip_longest

from numpy import typing as npt
from typing import Collection
from typing import Generator
from typing import Sequence
from typing import cast
from typing import Union
from typing import TypeVar
from typing import Iterable

from ch5mpy import Dataset
from ch5mpy._typing import SELECTOR
from ch5mpy.h5array.slice import FullSlice
from ch5mpy.utils import is_sequence

# ====================================================
# code
_DT = TypeVar("_DT", bound=np.generic)


def _get_array_sel(
    sel: tuple[Collection[int] | FullSlice, ...], index: int
) -> tuple[int | slice]:
    return cast(
        tuple[Union[int, slice]],
        tuple(slice(None) if isinstance(s, (Collection, FullSlice)) else index for s in sel)
    )


def _index_non_slice(elements: Iterable[Collection[int] | FullSlice]) -> int | None:
    for i, e in enumerate(elements):
        if not isinstance(e, FullSlice):
            return i

    return None


def _map_slice(
    elements: tuple[int | Sequence[int] | FullSlice | slice, ...]
) -> tuple[int | Sequence[int] | slice, ...]:
    return tuple(e.as_slice() if isinstance(e, FullSlice) else e for e in elements)


def _selection_iter(sel: tuple[Sequence[int] | FullSlice, ...]) -> Generator[
    tuple[tuple[int | Sequence[int] | slice, ...], tuple[int | slice, ...]], None, None
]:
    cut_index = _index_non_slice(sel)

    if len(sel) == 1 or cut_index is None:
        yield _map_slice(sel), _get_array_sel(sel, 0)

    else:
        pre = sel[:cut_index]
        post = sel[cut_index + 1:]
        cut = cast(Sequence[int], sel[cut_index])

        if len(cut) == 1:
            for p, arr_idx in _selection_iter(post):
                s = cut.start if isinstance(cut, FullSlice) else cut[0]

                yield _map_slice((*pre, s, *p)), (
                    *_get_array_sel(pre, 0),
                    *arr_idx,
                )

        else:
            for i, s in enumerate(cut):
                for p, arr_idx in _selection_iter(post):
                    yield _map_slice((*pre, s, *p)), (
                        *_get_array_sel(pre, 0),
                        i,
                        *arr_idx,
                    )


def read_from_dataset(dataset: Dataset[_DT],
                      selection: tuple[Sequence[int] | FullSlice, ...],
                      loading_array: npt.NDArray[_DT]) -> None:
    for dataset_idx, array_idx in _selection_iter(selection):
        dataset.read_direct(loading_array, source_sel=dataset_idx, dest_sel=array_idx)


def write_to_dataset(
        dataset: Dataset[_DT],
        values: npt.NDArray[_DT],
        selection: tuple[Sequence[int] | FullSlice, ...]
) -> None:
    if values.ndim == 0:
        values = values.reshape(1)

    for dataset_idx, array_idx in _selection_iter(selection):
        dataset.write_direct(values, source_sel=array_idx, dest_sel=dataset_idx)


def _check_bounds(max_: int, sel: Collection[int] | FullSlice, axis: int) -> None:
    valid = True

    if isinstance(sel, FullSlice):
        if sel.start < -max_ or sel.true_stop > max_:
            valid = False

    elif is_sequence(sel):
        if min(sel) < -max_ or max(sel) > max_:
            valid = False

    if not valid:
        raise IndexError(
            f"Index {sel} is out of bounds for axis {axis} with size {max_}."
        )


def _expand_index(index: tuple[SELECTOR, ...], shape: tuple[int, ...]) -> tuple[SELECTOR, ...]:
    exp_index: tuple[SELECTOR, ...] = tuple()
    i = 0

    for s in shape:
        if s == 1:
            exp_index += (0,)

        else:
            exp_index += (index[i],)
            i += 1
            if i == len(index):
                break

    return exp_index


def parse_selector(shape: tuple[int, ...],
                   index: SELECTOR | tuple[SELECTOR, ...]) \
        -> tuple[tuple[Sequence[int] | FullSlice, ...] | None, np.int_]:
    if index in (slice(None), ()):
        return None, np.product(shape)

    elif not isinstance(index, tuple):
        index = (index,)

    selection: tuple[Sequence[int] | FullSlice, ...] = ()

    for axis, (axis_len, axis_index) in enumerate(
        zip_longest(shape, _expand_index(index, shape), fillvalue=slice(None))
    ):
        if isinstance(axis_len, slice):
            raise RuntimeError

        parsed_axis_index: Union[FullSlice, Sequence[int]]

        if isinstance(axis_index, (slice, range)):
            parsed_axis_index = FullSlice(
                axis_index.start, axis_index.stop, axis_index.step, axis_len
            )

        elif is_sequence(axis_index):
            parsed_axis_index = axis_index

        elif isinstance(axis_index, int):
            parsed_axis_index = [axis_index]

        else:
            raise RuntimeError

        _check_bounds(axis_len, parsed_axis_index, axis)
        selection += (parsed_axis_index,)

    return selection, np.product([len(s) for s in selection])
