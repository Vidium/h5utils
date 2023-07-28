from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from ch5mpy.indexing.base import Indexer


class SingleIndex(Indexer):
    __slots__ = "_index", "_max"

    # region magic methods
    def __init__(self, index: int, max: int):
        if max <= index:
            raise IndexError(f"Selection {index} is out of bounds for axis with size {max}.")

        self._index = index
        self._max = max

    def __repr__(self) -> str:
        return f"SingleIndex({self._index} | {self._max})"

    def __len__(self) -> int:
        return 1

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SingleIndex):
            return False

        return self._index == other._index

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        return np.array(self._index, dtype=None)

    # endregion

    # region attributes
    @property
    def ndim(self) -> Literal[0]:
        return 0

    @property
    def is_whole_axis(self) -> bool:
        return self._max == 1

    # endregion

    # region methods
    def as_numpy_index(self) -> int:
        return self._index

    def as_slice(self) -> slice:
        return slice(self._index, self._index + 1)

    # endregion
