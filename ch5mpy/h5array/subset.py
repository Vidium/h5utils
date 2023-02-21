# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import Generator

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import TypeVar
from ch5mpy._typing import SELECTOR
from ch5mpy._typing import NP_FUNC

import ch5mpy.h5array.h5array as h5array
from ch5mpy import Dataset
from ch5mpy.h5array.indexing.selection import Selection
from ch5mpy.h5array.io import parse_selector
from ch5mpy.h5array.io import read_from_dataset
from ch5mpy.h5array.io import write_to_dataset
from ch5mpy.h5array.indexing.slice import FullSlice
from ch5mpy.h5array.indexing.slice import map_slice
from ch5mpy.objects.dataset import DatasetWrapper

# ====================================================
# code
_T = TypeVar("_T", bound=np.generic)
_DT = TypeVar("_DT", bound=np.generic)


class H5ArrayView(h5array.H5Array[_T]):
    """A view on a H5Array."""

    # region magic methods
    def __init__(self,
                 dset: Dataset[_T] | DatasetWrapper[_T],
                 sel: Selection):
        super().__init__(dset)
        self._selection = sel

    def __getitem__(self, index: SELECTOR | tuple[SELECTOR, ...]) -> _T | H5ArrayView[_T]:
        selection, nb_elements = parse_selector(self.shape_selection, index)

        if selection is None:
            return H5ArrayView(dset=self._dset, sel=self._selection)

        selection = selection.cast_on(self._selection)

        if nb_elements == 1:
            loading_array = np.empty((1,) * selection.max_ndim, dtype=self.dtype)
            read_from_dataset(self._dset, selection, loading_array)

            return cast(_T, loading_array[0])

        return H5ArrayView(dset=self._dset, sel=selection)

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
            self._dset[self._selection.get()] = value_arr

        else:
            write_to_dataset(self._dset, value_arr,  selection.cast_on(self._selection))

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
        return tuple(len(axis_sel) for axis_sel in self._selection if len(axis_sel) > 1 or axis_sel.ndim > 1)

    # endregion

    # region methods
    def read_direct(self,
                    dest: npt.NDArray[_T],
                    source_sel: tuple[FullSlice, ...],
                    dest_sel: tuple[FullSlice, ...]) -> None:
        source_sel_expanded = Selection(_expanded_selection(source_sel, self.shape_selection))
        read_from_dataset(self._dset,
                          source_sel_expanded.cast_on(self._selection),
                          dest[map_slice(dest_sel)])

    # endregion


def _expanded_selection(selection: tuple[FullSlice, ...],
                        shape: tuple[int, ...]) -> Generator[FullSlice, None, None]:
    sel_index = 0

    for s in shape:
        if s == 1:
            yield FullSlice.whole_axis(1)

        elif sel_index >= len(selection):
            yield FullSlice.whole_axis(s)

        else:
            yield selection[sel_index]
            sel_index += 1
