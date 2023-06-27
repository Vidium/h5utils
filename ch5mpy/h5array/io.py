# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import Any, Generator, TypeVar, cast

import numpy as np
from numpy import typing as npt

from ch5mpy.indexing.selection import Selection
from ch5mpy.objects.dataset import Dataset, DatasetWrapper

# ====================================================
# code
_DT = TypeVar("_DT", bound=np.generic)


class IterWithValue:
    def __init__(self, gen: Generator[Any, Any, Any]):
        self.gen = gen
        self.value = None

    def __iter__(self) -> Generator[Any, Any, Any]:
        self.value = yield from self.gen


def read_from_dataset(
    dataset: Dataset[_DT] | DatasetWrapper[_DT],
    selection: Selection,
    loading_array: npt.NDArray[_DT],
) -> None:
    if not dataset.size:
        if loading_array.size:
            raise ValueError("Reading from empty dataset.")
        return

    if not loading_array.size:
        return

    sel_iter = IterWithValue(selection.iter_h5(loading_array.shape))
    for dataset_idx, array_idx in sel_iter:
        # TODO : would be nice to be able to pass an array with random order in `dest_sel`
        dataset.read_direct(loading_array, source_sel=dataset_idx, dest_sel=array_idx)

    # reorder values in loading_array if needed
    if loading_array.ndim > 0:
        loading_array[:] = loading_array[sel_iter.value]


def read_one_from_dataset(
    dataset: Dataset[_DT] | DatasetWrapper[_DT],
    selection: Selection,
    dtype: np.dtype[_DT],
) -> _DT:
    loading_array = np.empty((), dtype=dtype)
    read_from_dataset(dataset, selection, loading_array)
    return cast(_DT, loading_array[()])


def write_to_dataset(
    dataset: Dataset[_DT] | DatasetWrapper[_DT],
    values: npt.NDArray[_DT],
    selection: Selection,
) -> None:
    selection_shape = selection.compute_shape(dataset.shape)

    if values.size == np.prod(selection_shape) and values.shape != selection_shape:
        values = values.reshape(selection_shape)

    for dataset_idx, array_idx in selection.iter_h5(values.shape):
        dataset.write_direct(values, source_sel=array_idx, dest_sel=dataset_idx)
