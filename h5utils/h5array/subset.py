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
from h5utils._typing import SELECTOR
from h5utils._typing import NP_FUNCTION

import h5utils.h5array.h5array as h5array
from h5utils import Dataset
from h5utils.h5array.io import map_slice
from h5utils.h5array.io import parse_selector
from h5utils.h5array.io import read_from_dataset
from h5utils.h5array.io import write_to_dataset
from h5utils.h5array.slice import FullSlice


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
            return cast(_T,
                        read_from_dataset(self._dset, _cast_selection(selection, on=self._selection),
                                          shape=(1,), dtype=None)[0])

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
            self._dset[map_slice(self._selection)] = value_arr

        else:
            write_to_dataset(self._dset, value_arr,  _cast_selection(selection, on=self._selection))

    def __len__(self) -> int:
        return self.shape[0]

    def __contains__(self, item: Any) -> bool:
        raise NotImplementedError

    def _inplace_operation(self, func: NP_FUNCTION, value: Any) -> H5ArrayView[_T]:
        raise NotImplementedError

    # endregion

    # region interface
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        return read_from_dataset(self._dset, self._selection, self.shape, dtype=dtype)

    # endregion

    # region attributes
    @property
    def shape_selection(self) -> tuple[int, ...]:
        return tuple(len(axis_sel) for axis_sel in self._selection)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(len(axis_sel) for axis_sel in self._selection if len(axis_sel) > 1)

    # endregion
