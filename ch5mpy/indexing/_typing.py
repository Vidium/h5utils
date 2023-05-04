# coding: utf-8
from __future__ import annotations

from typing import Union

from ch5mpy.indexing.list import ListIndex
from ch5mpy.indexing.slice import FullSlice
from ch5mpy.indexing.special import NewAxisType

# ====================================================
# imports

# ====================================================
# code
SELECTION_ELEMENT = Union[ListIndex, FullSlice, NewAxisType]
