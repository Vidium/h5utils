# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ch5mpy.h5array.indexing.slice import FullSlice


# ====================================================
# code
class ListIndex:

    # region magic methods
    def __init__(self,
                 elements: npt.NDArray[np.int_ | np.bool_]):
        self._elements = cast(npt.NDArray[np.int_],
                              np.where(elements.flatten())[0] if elements.dtype == bool else elements.flatten())
        self._ndim = 1 if elements.dtype == bool else elements.ndim

    def __repr__(self) -> str:
        return f"ListIndex({self._elements} | ndim={self._ndim})"

    def __getitem__(self, item: ListIndex | FullSlice) -> ListIndex:
        return ListIndex(self._elements[item])

    def __len__(self) -> int:
        return len(self._elements)

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        if dtype is None:
            return self._elements

        return self._elements.astype(dtype)

    # endregion

    # region attributes
    @property
    def ndim(self) -> int:
        return self._ndim

    @property
    def min(self) -> int:
        return int(self._elements.min())

    @property
    def max(self) -> int:
        return int(self._elements.max())

    # endregion

    # region methods
    def as_array(self) -> npt.NDArray[np.int_]:
        return self._elements

    # endregion
