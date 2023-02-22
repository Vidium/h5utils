# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import Iterable
from typing import Iterator
from typing import overload
from typing import Generator

from ch5mpy.h5array.indexing.list import ListIndex
from ch5mpy.h5array.indexing.slice import FullSlice


# ====================================================
# code
def _cast_one(s: ListIndex | FullSlice | None,
              o: ListIndex | FullSlice) -> ListIndex | FullSlice:
    if s is None:
        return o

    if isinstance(s, FullSlice) and s.is_whole_axis(o.max if isinstance(o, FullSlice) else s.max):
        return o

    elif isinstance(o, FullSlice) and o.is_whole_axis():
        return s

    return o[s]


def _iter_where(it: Iterable[Any], where: npt.NDArray[np.bool_], replace_by: Any) -> Generator[Any, None, None]:
    iterating = iter(it)

    for w in where:
        if w:
            yield next(iterating)

        else:
            yield replace_by


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
    def ndims(self) -> npt.NDArray[np.int_]:
        return np.array([i.ndim for i in self._indices])

    @property
    def max_ndim(self) -> int:
        return int(self.ndims.max())

    @property
    def shape(self) -> tuple[int, ...]:
        base_shape = tuple(len(i) for i in self._indices if len(i) > 1)
        return (1,) * (self.max_ndim - len(base_shape)) + base_shape

    @property
    def full_shape(self) -> tuple[int, ...]:
        return tuple(len(i) for i in self._indices)

    # endregion

    # region methods
    def get(self) -> tuple[npt.NDArray[np.int_] | slice, ...]:
        return tuple(i.as_slice() if isinstance(i, FullSlice) else i.as_array()
                     for i in self._indices)

    def cast_on(self, selection: Selection) -> Selection:
        return Selection((_cast_one(s, o) for s, o in zip(
            _iter_where(self, selection.ndims > 0, replace_by=None),
            selection
        )))

    # endregion
