# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import Collection

import numpy as np

from typing import Any
from typing import Union
from typing import Callable

# ====================================================
# code
SELECTOR = Union[int, bool, slice, range, Collection[int], Collection[bool], Collection[np.bool_], tuple[()]]

NP_FUNC = Callable[..., Any]
H5_FUNC = Callable[..., Any]
