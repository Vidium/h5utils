from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pathlib import Path
from functools import partial
from typing import Any

import ch5mpy
from ch5mpy.functions.creation_routines import ArrayCreationFunc
from ch5mpy.indexing import map_slice


class ArrayCreationFuncRandom(ArrayCreationFunc):
    # region magic methods
    def __init__(self, name: str, random_func: Any):
        super().__init__(name)
        self._random_func = random_func

    def __call__(  # type: ignore[override]
        self,
        *dims: int,
        name: str,
        loc: str | Path | ch5mpy.File | ch5mpy.Group,
        dtype: npt.DTypeLike = np.float64,
        chunks: bool | tuple[int, ...] = True,
        maxshape: int | tuple[int | None, ...] | None = None,
    ) -> ch5mpy.H5Array[Any]:
        arr = super().__call__(dims, None, name, loc, dtype, chunks, maxshape)

        for index, chunk in arr.iter_chunks():
            chunk = self._random_func(*chunk.shape)

            arr.dset.write_direct(
                chunk,
                source_sel=map_slice(index, shift_to_zero=True),
                dest_sel=map_slice(index),
            )

        return arr

    # endregion

    # region methods
    def anonymous(  # type: ignore[override]
        self,
        *dims: int,
        dtype: npt.DTypeLike = np.float64,
        chunks: bool | tuple[int, ...] = True,
        maxshape: int | tuple[int | None, ...] | None = None,
    ) -> partial[ch5mpy.H5Array[Any]]:
        return super().anonymous(dims, None, dtype, chunks, maxshape)

    # endregion


rand = ArrayCreationFuncRandom("rand", np.random.rand)
