from __future__ import annotations

from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

import ch5mpy
import ch5mpy.indexing as ci
from ch5mpy import Dataset
from ch5mpy._typing import NP_FUNC, SELECTOR
from ch5mpy.h5array.array import as_array
from ch5mpy.h5array.io import read_from_dataset, read_one_from_dataset, write_to_dataset
from ch5mpy.objects import DatasetWrapper


_T = TypeVar("_T", bound=np.generic)


class H5ArrayView(ch5mpy.H5Array[_T]):
    """A view on a H5Array."""

    # region magic methods
    def __init__(self, dset: Dataset[_T] | DatasetWrapper[_T], sel: ci.Selection):
        super().__init__(dset)
        self._selection = sel

    def __getitem__(self, index: SELECTOR | tuple[SELECTOR, ...]) -> _T | ch5mpy.H5Array[_T]:
        selection = ci.Selection.from_selector(index, self.shape)

        if selection.is_empty:
            return H5ArrayView(dset=self._dset, sel=self._selection)

        selection = selection.cast_on(self._selection)

        if selection.is_empty:
            return ch5mpy.H5Array(self._dset)

        if selection.out_shape == ():
            return read_one_from_dataset(self._dset, selection, self.dtype)

        return H5ArrayView(dset=self._dset, sel=selection)

    def __setitem__(self, index: SELECTOR | tuple[SELECTOR, ...], value: Any) -> None:
        selection = ci.Selection.from_selector(index, self.shape)
        write_to_dataset(self._dset, as_array(value, self.dtype), selection.cast_on(self._selection))

    def __len__(self) -> int:
        return self.shape[0]

    def _inplace(self, func: NP_FUNC, value: Any) -> H5ArrayView[_T]:
        if np.issubdtype(self.dtype, str):
            raise TypeError("Cannot perform inplace operation on str H5Array.")

        # special case : 0D array
        if self.shape == ():
            self._dset[:] = func(self._dset[:], value)
            return self

        # general case : 1D+ array
        for index, chunk in self.iter_chunks():
            func(chunk, value, out=chunk)

            for dest_sel, source_sel in ci.Selection(index, self.shape).cast_on(self._selection).iter_indexers():
                self._dset.write_direct(chunk, source_sel=source_sel, dest_sel=dest_sel)

        return self

    # endregion

    # region interface
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        loading_array = np.empty(
            self._selection.out_shape,
            dtype or self.dtype,
        )
        read_from_dataset(self._dset, self._selection, loading_array)

        return loading_array.reshape(self.shape)

    # endregion

    # region attributes
    @property
    def shape(self) -> tuple[int, ...]:
        return self._selection.out_shape

    # endregion

    # region methods
    def astype(self, dtype: npt.DTypeLike, inplace: bool = False) -> H5ArrayView[Any]:
        """
        Cast an H5Array to a specified dtype.
        This does not perform a copy, it returns a wrapper around the underlying H5 dataset.
        """
        if inplace:
            raise TypeError("Cannot cast inplace a view of an H5Array.")

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

    def read_direct(
        self,
        dest: npt.NDArray[_T],
        source_sel: tuple[int | slice, ...],
        dest_sel: tuple[int | slice, ...],
    ) -> None:
        dataset = self._dset.asstr() if np.issubdtype(self.dtype, str) else self._dset
        read_from_dataset(
            dataset,
            ci.Selection(
                (ci.as_indexer(sl, max=axis_shape) for sl, axis_shape in zip(source_sel, self.shape)),
                shape=self.shape,
            ).cast_on(self._selection),
            dest[dest_sel],
        )

    # endregion
