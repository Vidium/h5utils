# coding: utf-8
from __future__ import annotations

from typing import Union

from ch5mpy.h5array.indexing.list import ListIndex
from ch5mpy.h5array.indexing.slice import FullSlice
from ch5mpy.h5array.indexing.special import NewAxisType

# ====================================================
# imports

# ====================================================
# code
SELECTION_ELEMENT = Union[ListIndex, FullSlice, NewAxisType]
