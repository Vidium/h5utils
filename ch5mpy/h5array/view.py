# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import TypeVar
from ch5mpy._typing import SELECTOR
from ch5mpy._typing import NP_FUNC

import ch5mpy
from ch5mpy import Dataset
from ch5mpy.h5array.indexing.selection import Selection
from ch5mpy.h5array.io import read_from_dataset
from ch5mpy.h5array.io import read_one_from_dataset
from ch5mpy.h5array.io import write_to_dataset
from ch5mpy.h5array.indexing.slice import FullSlice
from ch5mpy.objects.dataset import DatasetWrapper

# ====================================================
# code
_T = TypeVar("_T", bound=np.generic)
_OT = TypeVar("_OT")


class H5ArrayView(ch5mpy.H5Array[_T]):
    """A view on a H5Array."""

    # region magic methods
    def __init__(self,
                 dset: Dataset[_T] | DatasetWrapper[_T],
                 sel: Selection):
        super().__init__(dset)
        self._selection = sel

    def __getitem__(self, index: SELECTOR | tuple[SELECTOR, ...]) -> _T | H5ArrayView[_T]:
        selection = Selection.from_selector(index, self.shape)

        if selection.is_empty:
            return H5ArrayView(dset=self._dset, sel=self._selection)

        selection = selection.cast_on(self._selection)

        if selection.compute_shape(self.shape) == ():
            return read_one_from_dataset(self._dset, selection, self.dtype)

        return H5ArrayView(dset=self._dset, sel=selection)

    def __setitem__(self, index: SELECTOR | tuple[SELECTOR, ...], value: Any) -> None:
        selection = Selection.from_selector(index, self.shape)

        try:
            value_arr = np.array(value, dtype=self.dtype)

        except ValueError:
            raise ValueError(f'Could set value of type {type(value)} in H5Array of type {self.dtype}.')

        if np.product(selection.compute_shape(self.shape)) != value_arr.size:
            raise ValueError(f"{' x '.join(map(str, self.shape if selection.is_empty else map(len, selection)))} "
                             f"values were selected but {' x '.join(map(str, value_arr.shape))} were given.")

        if selection.is_empty:
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
        loading_array = np.empty(self.shape, dtype or self.dtype)
        read_from_dataset(self._dset, self._selection, loading_array)

        return loading_array

    # endregion

    # region attributes
    @property
    def shape(self) -> tuple[int, ...]:
        return self._selection.compute_shape(self._dset.shape)

    # endregion

    # region methods
    def astype(self, dtype: npt.DTypeLike) -> H5ArrayView[Any]:
        """
        Cast an H5Array to a specified dtype.
        This does not perform a copy, it returns a wrapper around the underlying H5 dataset.
        """
        if np.issubdtype(dtype, str) and (np.issubdtype(self._dset.dtype, str) or self._dset.dtype == object):
            return H5ArrayView(self._dset.asstr(), sel=self._selection)

        return H5ArrayView(self._dset.astype(dtype), sel=self._selection)

    def maptype(self, otype: type[Any]) -> H5ArrayView[Any]:
        """
        Cast an H5Array to any object type.
        This extends H5Array.astype() to any type <T>, where it is required that an object <T> can be constructed as
        T(v) for any value <v> in the dataset.
        """
        return H5ArrayView(self._dset.maptype(otype), sel=self._selection)

    def read_direct(self,
                    dest: npt.NDArray[_T],
                    source_sel: tuple[slice, ...],
                    dest_sel: tuple[slice, ...]) -> None:
        dset = self._dset.asstr() if np.issubdtype(self.dtype, str) else self._dset
        read_from_dataset(dset,
                          Selection((FullSlice.from_slice(s) for s in source_sel)).cast_on(self._selection),
                          dest[dest_sel])

    # endregion
