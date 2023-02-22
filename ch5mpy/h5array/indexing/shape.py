# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from dataclasses import dataclass

from typing import Iterator
from typing import Iterable

from ch5mpy.h5array.indexing.selection import Selection


# ====================================================
# code
@dataclass
class ShapeElement:
    size: int
    ndim: int = 1


class DimShape:
    def __init__(self, shape: Iterable[ShapeElement]):
        self._shape = tuple(shape)

    def __repr__(self) -> str:
        return repr(self._shape)

    def __iter__(self) -> Iterator[ShapeElement]:
        return iter(s for s in self._shape if s.ndim > 0)

    @classmethod
    def from_selection(cls, selection: Selection) -> DimShape:
        return DimShape(ShapeElement(len(s), ndim=s.ndim) for s in selection)

    @classmethod
    def from_shape(cls, shape: tuple[int, ...]) -> DimShape:
        return DimShape(ShapeElement(s, ndim=1) for s in shape)
