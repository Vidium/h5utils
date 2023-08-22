from __future__ import annotations

import numpy as np

from numpy import typing as npt
from typing import Any


def _as_valid_dtype(arr: npt.NDArray[Any], dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    if np.issubdtype(dtype, str):
        return arr.astype(str)

    return arr
