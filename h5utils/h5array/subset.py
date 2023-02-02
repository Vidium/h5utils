# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from itertools import zip_longest

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import Union
from typing import TypeVar
from typing import Iterable
from typing import Sequence
from typing import Generator
from typing import Collection
from h5utils._typing import SELECTOR
from h5utils._typing import NP_FUNCTION

import h5utils as hu
from h5utils import Dataset
from h5utils.h5array.slice import FullSlice
from h5utils.utils import is_sequence


# ====================================================
# code
_T = TypeVar("_T", bound=np.generic)
_DT = TypeVar("_DT", bound=np.generic)


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


def parse_selector(shape: tuple[int, ...],
                   index: SELECTOR | tuple[SELECTOR]) \
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


def _cast_selection(
        selection: tuple[Sequence[int] | FullSlice, ...],
        *,
        on: tuple[Sequence[int] | FullSlice, ...],
) -> tuple[Sequence[int] | FullSlice, ...]:
    casted_indices: tuple[Sequence[int] | FullSlice, ...] = ()

    for s, o in zip(selection, on):
        if isinstance(s, FullSlice) and \
                s.is_whole_axis(o.max if isinstance(o, FullSlice) else s.max):
            casted_indices += (o,)

        elif isinstance(o, FullSlice) and o.is_whole_axis():
            casted_indices += (s,)

        else:
            casted_indices += (np.array(o)[np.array(s)],)

    return casted_indices


def _map_slice(
    elements: tuple[int | Collection[int] | FullSlice | slice, ...]
) -> tuple[int | Collection[int] | slice, ...]:
    return tuple(e.as_slice() if isinstance(e, FullSlice) else e for e in elements)


def _index_non_slice(elements: Iterable[Collection[int] | FullSlice]) -> int | None:
    for i, e in enumerate(elements):
        if not isinstance(e, FullSlice):
            return i

    return None


def _get_array_sel(
    sel: tuple[Collection[int] | FullSlice, ...], index: int
) -> tuple[int | slice]:
    return cast(
        tuple[Union[int, slice]],
        tuple(slice(None) if isinstance(s, (Collection, FullSlice)) else index for s in sel)
    )


def _selection_iter(sel: tuple[Sequence[int] | FullSlice, ...]) -> Generator[
    tuple[tuple[int | Collection[int] | slice, ...], tuple[int | slice, ...]], None, None
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


def load_array(
    dataset: Dataset[_DT],
    selection: tuple[Sequence[int] | FullSlice, ...],
    shape: tuple[int, ...],
    dtype: _DT | None,
) -> npt.NDArray[_DT]:
    if dtype is None:
        dtype = dataset.dtype                                                                 # type: ignore[assignment]

    loaded_array = np.empty(shape, dtype=dtype)

    for dataset_idx, array_idx in _selection_iter(selection):
        dataset.read_direct(loaded_array, source_sel=dataset_idx, dest_sel=array_idx)

    return loaded_array


class H5ArrayView(hu.H5Array[_T]):
    """A view on a H5Array."""

    # region magic methods
    def __init__(self, dset: Dataset[_T], sel: tuple[Sequence[int] | FullSlice, ...]):
        super().__init__(dset)
        self._selection = sel

    def __getitem__(self, index: SELECTOR | tuple[SELECTOR]) -> _T | H5ArrayView[_T]:
        selection, nb_elements = parse_selector(self.shape_selection, index)

        if selection is None:
            return H5ArrayView(dset=self._dset, sel=self._selection)

        elif nb_elements == 1:
            return cast(_T,
                        load_array(self._dset, _cast_selection(selection, on=self._selection), shape=(1,),
                                   dtype=None)[0])

        return H5ArrayView(dset=self._dset, sel=_cast_selection(selection, on=self._selection))

    def __len__(self) -> int:
        return self.shape[0]

    def __contains__(self, item: Any) -> bool:
        raise NotImplementedError

    def _inplace_operation(self, func: NP_FUNCTION, value: Any) -> H5ArrayView[_T]:
        raise NotImplementedError

    # endregion

    # region interface
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        return load_array(self._dset, self._selection, self.shape, dtype=dtype)

    # endregion

    # region attributes
    @property
    def shape_selection(self) -> tuple[int, ...]:
        return tuple(len(axis_sel) for axis_sel in self._selection)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis_sel) for axis_sel in self._selection if len(axis_sel) > 1)

    # endregion
