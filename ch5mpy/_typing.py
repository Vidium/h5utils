# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import Any
from typing import Union
from typing import Iterable
from typing import Callable

# ====================================================
# code
SELECTOR = Union[int, bool, slice, range, Iterable[Any], tuple[()]]

NP_FUNC = Callable[..., Any]
H5_FUNC = Callable[..., Any]
