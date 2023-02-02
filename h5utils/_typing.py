# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from numbers import Number

import numpy.typing as npt
from typing import Any
from typing import Union
from typing import Protocol
from typing import Sequence


# ====================================================
# code
SELECTOR = Union[int, range, slice, Sequence[int], tuple[()]]


class NP_FUNCTION(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> npt.NDArray[Any] | Number | bool:
        ...
