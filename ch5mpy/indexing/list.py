# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ch5mpy.indexing._typing import SELECTION_ELEMENT


# ====================================================
# code
class ListIndex:
    # region magic methods
    def __init__(self, elements: npt.NDArray[np.int_]):
        if elements.dtype != int:
            raise RuntimeError

        self._elements = elements

    def __repr__(self) -> str:
        flat_elements_repr = str(self._elements).replace("\n", "")
        return f"ListIndex({flat_elements_repr} | ndim={self.ndim})"

    def __getitem__(self, item: SELECTION_ELEMENT | tuple[SELECTION_ELEMENT, ...]) -> ListIndex:
        from ch5mpy.indexing.special import NewAxisType

        if isinstance(item, tuple):
            if any(isinstance(e, NewAxisType) for e in item):
                raise RuntimeError

            casted_items: tuple[slice | npt.NDArray[np.int_], ...] = tuple(
                i.as_array() if isinstance(i, ListIndex) else i.as_slice() for i in item  # type: ignore[union-attr]
            )

            return ListIndex(self._elements[casted_items])

        if isinstance(item, NewAxisType):
            raise RuntimeError

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

    @property
    def is_zero(self) -> bool:
        return (self._elements.ndim == 0 and self._elements == 0) or (
            self._elements.shape == (1,) and all(self._elements == 0)
        )

    # endregion

    # region methods
    def as_array(self, sorted: bool = False) -> npt.NDArray[np.int_]:
        if sorted:
            return np.sort(self._elements)
        return self._elements

    def squeeze(self) -> ListIndex:
        return ListIndex(np.squeeze(self._elements))

    def expand(self, n: int) -> ListIndex:
        if n < self.ndim:
            raise RuntimeError

        expanded_shape = (1,) * (n - self.ndim) + self.shape
        return ListIndex(self._elements.reshape(expanded_shape))

    def broadcast_to(self, shape: tuple[int, ...]) -> ListIndex:
        return ListIndex(np.broadcast_to(self._elements, shape))

    # endregion
