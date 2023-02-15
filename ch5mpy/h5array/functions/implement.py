# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import importlib
import numpy as np

from typing import Callable

from ch5mpy._typing import NP_FUNC
from ch5mpy._typing import H5_FUNC


# ====================================================
# code
HANDLED_FUNCTIONS: dict[NP_FUNC | np.ufunc, H5_FUNC] = {}


def implements(np_function: NP_FUNC | np.ufunc) -> Callable[[H5_FUNC], H5_FUNC]:
    """Register an __array_function__ implementation for DiagonalArray objects."""

    def decorator(func: H5_FUNC) -> H5_FUNC:
        HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


# manually import function implementations otherwise they are never imported
importlib.__import__("ch5mpy.h5array.functions.two_arrays")
importlib.__import__("ch5mpy.h5array.functions.element_wise")
