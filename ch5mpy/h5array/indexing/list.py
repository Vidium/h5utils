# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ch5mpy.h5array.indexing.slice import FullSlice


# ====================================================
# code
class ListIndex:

    # region magic methods
    def __init__(self,
                 elements: npt.NDArray[np.int_]):
        if elements.dtype != int:
            raise RuntimeError

        self._elements = elements

    def __repr__(self) -> str:
        flat_elements_repr = str(self._elements).replace('\n', '')
        return f"ListIndex({flat_elements_repr} | ndim={self.ndim})"

    def __getitem__(self, item: ListIndex | FullSlice | tuple[ListIndex | FullSlice, ...]) -> ListIndex:
        if isinstance(item, tuple):
            casted_items: tuple[slice | npt.NDArray[np.int_], ...] = \
                tuple(i.as_array() if isinstance(i, ListIndex) else i.as_slice() for i in item)
            return ListIndex(self._elements[casted_items])

        return ListIndex(self._elements[item])

    def __len__(self) -> int:
        if self._elements.ndim == 0:
            return 1

        return len(self._elements)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ListIndex):
            return False

        return np.array_equal(self._elements, other._elements)

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        if dtype is None:
            return self._elements

        return self._elements.astype(dtype)

    # endregion

    # region attributes
    @property
    def ndim(self) -> int:
        return self._elements.ndim

    @property
    def min(self) -> int:
        return int(self._elements.min())

    @property
    def max(self) -> int:
        return int(self._elements.max())

    @property
    def shape(self) -> tuple[int, ...]:
        return self._elements.shape

    @property
    def size(self) -> int:
        return self._elements.size

    # endregion

    # region methods
    def as_array(self) -> npt.NDArray[np.int_]:
        return self._elements

    def squeeze(self) -> ListIndex:
        return ListIndex(np.squeeze(self._elements))

    def expand(self, n: int) -> ListIndex:
        if n < self.ndim:
            raise RuntimeError

        expanded_shape = (1,) * (n - self.ndim) + self.shape
        return ListIndex(self._elements.reshape(expanded_shape))

    # endregion
