# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from numbers import Number

import numpy.typing as npt
from typing import Any


# ====================================================
# code
def arange_nd(shape: tuple[int, ...],
              start: Number | None = None,
              step: Number | None = None,
              dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
    start_ = 0 if start is None else start
    stop = np.product(shape) + start_                                                      # type: ignore[call-overload]

    return np.arange(start=start_, stop=stop, step=step, dtype=dtype).reshape(shape)
