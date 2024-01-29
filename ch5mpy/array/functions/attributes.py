from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Any

from ch5mpy.array.functions.implement import implements

if TYPE_CHECKING:
    from ch5mpy.array.array import H5Array


@implements(np.ndim)
def ndim(a: H5Array[Any]) -> int:
    return a.ndim
