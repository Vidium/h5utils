# coding: utf-8

# ====================================================
# imports
from dataclasses import dataclass

from ch5mpy.h5array.indexing.selection import Selection


# ====================================================
# code
@dataclass
class ShapeElement:
    size: int
    ndim: int = 1


def shape_dim_from_selection(shape: tuple[int, ...],
                             selection: Selection) -> tuple[ShapeElement, ...]:
    return tuple(ShapeElement(s, ndim=sel.ndim) for s, sel in zip(shape, selection))
