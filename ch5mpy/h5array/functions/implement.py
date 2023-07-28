from __future__ import annotations

import importlib
from functools import partial

from typing import Callable

from ch5mpy._typing import NP_FUNC
from ch5mpy._typing import H5_FUNC


HANDLED_FUNCTIONS: dict[NP_FUNC | NP_FUNC, H5_FUNC] = {}


def implements(*np_functions: NP_FUNC | NP_FUNC) -> Callable[[H5_FUNC], H5_FUNC]:
    """Register an __array_function__ implementation for H5Array objects."""

    def decorator(func: H5_FUNC) -> H5_FUNC:
        for f in np_functions:
            HANDLED_FUNCTIONS[f] = func
        return func

    return decorator


def register(function: partial[H5_FUNC], implements: NP_FUNC) -> None:
    HANDLED_FUNCTIONS[implements] = function


# manually import function implementations otherwise they are never imported
importlib.__import__("ch5mpy.h5array.functions.routines")
importlib.__import__("ch5mpy.h5array.functions.creation_routines")
importlib.__import__("ch5mpy.h5array.functions.two_arrays")
importlib.__import__("ch5mpy.h5array.functions.element_wise")
