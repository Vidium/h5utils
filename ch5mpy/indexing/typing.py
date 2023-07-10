# coding: utf-8
from typing import Union

from ch5mpy.indexing.list import ListIndex
from ch5mpy.indexing.slice import FullSlice
from ch5mpy.indexing.special import NewAxisType

SELECTION_ELEMENT = Union[ListIndex, FullSlice, NewAxisType]
