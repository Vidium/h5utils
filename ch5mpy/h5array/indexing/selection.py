# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Iterable
from typing import Iterator
from typing import overload

from ch5mpy.h5array.indexing.list import ListIndex
from ch5mpy.h5array.indexing.slice import FullSlice


# ====================================================
# code
def _cast_one(s: ListIndex | FullSlice,
              o: ListIndex | FullSlice) -> ListIndex | FullSlice:
    if isinstance(s, FullSlice) and s.is_whole_axis(o.max if isinstance(o, FullSlice) else s.max):
        return o

    elif isinstance(o, FullSlice) and o.is_whole_axis():
        return s

    return o[s]


class Selection:

    # region magic methods
    def __init__(self,
                 indices: Iterable[ListIndex | FullSlice]):
        self._indices = tuple(indices)

    def __repr__(self) -> str:
        return f"Selection{self._indices}"

    @overload
    def __getitem__(self, item: int) -> ListIndex | FullSlice: ...
    @overload
    def __getitem__(self, item: slice) -> Selection: ...
    def __getitem__(self, item: int | slice) -> ListIndex | FullSlice | Selection:
        if isinstance(item, int):
            return self._indices[item]

        return Selection(self._indices[item])

    def __iter__(self) -> Iterator[ListIndex | FullSlice]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    # endregion

    # region attributes
    @property
    def max_ndim(self) -> int:
        return max(i.ndim for i in self._indices)

    # endregion

    # region methods
    def get(self) -> tuple[npt.NDArray[np.int_] | slice, ...]:
        return tuple(i.as_slice() if isinstance(i, FullSlice) else i.as_array()
                     for i in self._indices)

    def cast_on(self, selection: Selection) -> Selection:
        return Selection((_cast_one(s, o) for s, o in zip(self, selection)))

    # endregion
