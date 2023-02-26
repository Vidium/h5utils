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
              dtype: npt.DTypeLike | None = None,
              like: npt.ArrayLike | None = None) -> npt.NDArray[Any]:
    if start is None:
        stop = np.product(shape)

    else:
        stop = np.product(shape) + start                                                   # type: ignore[call-overload]

    return np.arange(start=start, stop=stop, step=step, dtype=dtype, like=like            # type: ignore[misc, arg-type]
                     ).reshape(shape)
