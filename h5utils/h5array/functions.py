# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import TypeVar
from typing import Callable
from typing import TYPE_CHECKING
from h5utils._typing import NP_FUNCTION

import h5utils

if TYPE_CHECKING:
    from h5utils import H5Array
    from h5utils import Dataset


# ====================================================
# code
_T = TypeVar("_T", bound=np.generic, covariant=True)

_HANDLED_FUNCTIONS = {}


def implements(np_function: NP_FUNCTION) -> Callable[[NP_FUNCTION], NP_FUNCTION]:
    """Register an __array_function__ implementation for DiagonalArray objects."""

    def decorator(func: NP_FUNCTION) -> NP_FUNCTION:
        _HANDLED_FUNCTIONS[np_function] = func
        return func

    return decorator


# TODO : handle large arrays
@implements(np.sum)
def sum(arr: H5Array[_T], axis: int | None = None) -> _T | npt.NDArray[_T]:
    return np.sum(arr.dset, axis=axis)  # type: ignore[no-any-return, call-overload]


@implements(np.mean)
def mean(arr: H5Array[_T], axis: int | None = None) -> _T | npt.NDArray[_T]:
    return np.mean(arr.dset, axis=axis)  # type: ignore[no-any-return, call-overload]


@implements(np.array_equal)
def array_equal(
    arr1: H5Array[Any] | npt.NDArray[Any], arr2: H5Array[Any] | npt.NDArray[Any]
) -> bool:
    if isinstance(arr1, h5utils.H5Array):
        arr1_data: npt.NDArray[Any] | Dataset[Any] = np.array(arr1)
    else:
        arr1_data = arr1

    if isinstance(arr2, h5utils.H5Array):
        arr2_data: npt.NDArray[Any] | Dataset[Any] = np.array(arr2)
    else:
        arr2_data = arr2

    return np.array_equal(arr1_data, arr2_data)  # type: ignore[arg-type]
