# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from pathlib import Path
from functools import partial

import numpy.typing as npt
from typing import Any

import ch5mpy
from ch5mpy.write import _store_dataset

# ====================================================
# code
_ACF_NAME = {
    None: 'empty',
    0: 'zeros',
    1: 'ones'
}


class ArrayCreationFunc:

    # region magic methods
    def __init__(self,
                 fill_value: Any):
        self._fill_value = fill_value

    def __repr__(self) -> str:
        return f"<function ch5mpy.{_ACF_NAME.get(self._fill_value, 'full')} at {hex(id(self))}>"

    def __call__(self,
                 shape: int | tuple[int, ...],
                 name: str,
                 loc: str | Path | ch5mpy.File | ch5mpy.Group,
                 dtype: npt.DTypeLike = np.float64,
                 chunks: bool | tuple[int, ...] | None = None,
                 maxshape: int | tuple[int | None, ...] | None = None) -> ch5mpy.H5Array[Any]:
        shape = shape if isinstance(shape, tuple) else (shape,)

        if not isinstance(loc, (ch5mpy.File, ch5mpy.Group)):
            loc = ch5mpy.File(loc, mode=ch5mpy.H5Mode.READ_WRITE)

        dset = _store_dataset(loc, name, shape=shape, dtype=dtype, chunks=chunks, maxshape=maxshape,
                              fill_value=self._fill_value)

        return ch5mpy.H5Array(dset)

    # endregion

    # region methods
    def p(self,
          shape: int | tuple[int, ...],
          dtype: npt.DTypeLike = np.float64,
          chunks: bool | tuple[int, ...] | None = None,
          maxshape: int | tuple[int | None, ...] | None = None) -> partial[ch5mpy.H5Array[Any]]:
        return partial(self.__call__, shape=shape, dtype=dtype, chunks=chunks, maxshape=maxshape)

    # endregion


empty = ArrayCreationFunc(None)
zeros = ArrayCreationFunc(0)
ones = ArrayCreationFunc(1)


def full(shape: int | tuple[int, ...],
         fill_value: Any,
         name: str,
         loc: str | Path | ch5mpy.File | ch5mpy.Group,
         dtype: npt.DTypeLike = np.float64,
         chunks: bool | tuple[int, ...] | None = None,
         maxshape: int | tuple[int | None, ...] | None = None) -> ch5mpy.H5Array[Any]:
    return ArrayCreationFunc(fill_value)(shape, name, loc, dtype, chunks, maxshape)
