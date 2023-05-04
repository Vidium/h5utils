# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ch5mpy.indexing.list import ListIndex
from ch5mpy.indexing.slice import FullSlice

if TYPE_CHECKING:
    from ch5mpy.indexing._typing import SELECTION_ELEMENT


# ====================================================
# code
class NewAxisType:
    # region magic methods
    def __new__(cls) -> NewAxisType:
        return NewAxis

    def __repr__(self) -> str:
        return "<NewAxis>"

    def __getitem__(self, item: SELECTION_ELEMENT | tuple[SELECTION_ELEMENT, ...]) -> NewAxisType | None:
        if isinstance(item, tuple):
            if len(item) != 1:
                raise ValueError

            item = item[0]

        if isinstance(item, FullSlice) and item.is_whole_axis():
            return NewAxis

        elif isinstance(item, ListIndex) and item.is_zero:
            return None

        else:
            raise ValueError

    def __len__(self) -> int:
        return 1

    def __eq__(self, other: Any) -> bool:
        return other is NewAxis

    # endregion

    # region attributes
    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def ndim(self) -> int:
        return 1

    # endregion


NewAxis = object.__new__(NewAxisType)

Placeholder = object()
