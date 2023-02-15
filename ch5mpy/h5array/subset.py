# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import TypeVar
from typing import Sequence
from ch5mpy._typing import SELECTOR
from ch5mpy._typing import NP_FUNC

import ch5mpy.h5array.h5array as h5array
from ch5mpy import Dataset
from ch5mpy.h5array.io import parse_selector
from ch5mpy.h5array.io import read_from_dataset
from ch5mpy.h5array.io import write_to_dataset
from ch5mpy.h5array.slice import FullSlice
from ch5mpy.h5array.slice import map_slice

# ====================================================
# code
_T = TypeVar("_T", bound=np.generic)
_DT = TypeVar("_DT", bound=np.generic)


def _cast_selection(
        selection: tuple[Sequence[int] | FullSlice, ...],
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


class H5ArrayView(h5array.H5Array[_T]):
    """A view on a H5Array."""

    # region magic methods
    def __init__(self, dset: Dataset[_T], sel: tuple[Sequence[int] | FullSlice, ...]):
        super().__init__(dset)
        self._selection = sel

    def __getitem__(self, index: SELECTOR | tuple[SELECTOR, ...]) -> _T | H5ArrayView[_T]:
        selection, nb_elements = parse_selector(self.shape_selection, index)

        if selection is None:
            return H5ArrayView(dset=self._dset, sel=self._selection)

        elif nb_elements == 1:
            loading_array = np.empty(1, dtype=self.dtype)
            read_from_dataset(self._dset, _cast_selection(selection, on=self._selection), loading_array)

            return cast(_T, loading_array[0])

        return H5ArrayView(dset=self._dset, sel=_cast_selection(selection, on=self._selection))

    def __setitem__(self, index: SELECTOR | tuple[SELECTOR, ...], value: Any) -> None:
        selection, nb_elements = parse_selector(self.shape_selection, index)

        try:
            value_arr = np.array(value, dtype=self.dtype)

        except ValueError:
            raise ValueError(f'Could set value of type {type(value)} in H5Array of type {self.dtype}.')

        if nb_elements != value_arr.size:
            raise ValueError(f"{' x '.join(map(str, self.shape if selection is None else map(len, selection)))} "
                             f"values were selected but {' x '.join(map(str, value_arr.shape))} were given.")

        if selection is None:
            selection_as_slices: tuple[Sequence[int] | slice, ...] = \
                tuple(e.as_slice() if isinstance(e, FullSlice) else e for e in self._selection)
            self._dset[selection_as_slices] = value_arr

        else:
            write_to_dataset(self._dset, value_arr,  _cast_selection(selection, on=self._selection))

    def __len__(self) -> int:
        return self.shape[0]

    def __contains__(self, item: Any) -> bool:
        raise NotImplementedError

    def _inplace_operation(self, func: NP_FUNC, value: Any) -> H5ArrayView[_T]:
        raise NotImplementedError

    # endregion

    # region interface
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        if dtype is None:
            dtype = self.dtype

        loading_array = np.empty(self.shape, dtype)
        read_from_dataset(self._dset, self._selection, loading_array)

        return loading_array

    # endregion

    # region attributes
    @property
    def shape_selection(self) -> tuple[int, ...]:
        return tuple(len(axis_sel) for axis_sel in self._selection)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis_sel) for axis_sel in self._selection if len(axis_sel) > 1)

    # endregion

    # region methods
    def read_direct(self,
                    dest: npt.NDArray[_T],
                    source_sel: tuple[FullSlice, ...],
                    dest_sel: tuple[FullSlice, ...]) -> None:
        nb_whole_axes_before = (self._dset.ndim - len(source_sel))
        whole_axes_before = tuple(FullSlice.whole_axis(s) for s in self.shape[:nb_whole_axes_before])
        nb_whole_axes_after = (len(whole_axes_before) + len(source_sel))
        whole_axes_after = tuple(FullSlice.whole_axis(s) for s in self.shape[nb_whole_axes_after:])

        source_sel_casted = _cast_selection(whole_axes_before + source_sel + whole_axes_after, on=self._selection)
        read_from_dataset(self._dset, source_sel_casted, dest[map_slice(dest_sel)])

    # endregion
