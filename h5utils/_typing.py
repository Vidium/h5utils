# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import Any
from typing import Union
from typing import Sequence
from typing import Callable

# ====================================================
# code
SELECTOR = Union[int, range, slice, Sequence[int], tuple[()]]

NP_FUNC = Callable[..., Any]
H5_FUNC = Callable[..., Any]
