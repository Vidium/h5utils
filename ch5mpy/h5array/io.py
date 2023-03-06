# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

from numpy import typing as npt
from typing import cast
from typing import TypeVar

from ch5mpy import Dataset
from ch5mpy.h5array.indexing.selection import Selection
from ch5mpy.objects.dataset import DatasetWrapper

# ====================================================
# code
_DT = TypeVar("_DT", bound=np.generic)


def read_from_dataset(dataset: Dataset[_DT] | DatasetWrapper[_DT],
                      selection: Selection,
                      loading_array: npt.NDArray[_DT]) -> None:
    for dataset_idx, array_idx in selection.iter_h5(loading_array.shape):
        dataset.read_direct(loading_array, source_sel=dataset_idx, dest_sel=array_idx)


def read_one_from_dataset(dataset: Dataset[_DT] | DatasetWrapper[_DT],
                          selection: Selection,
                          dtype: np.dtype[_DT]) -> _DT:
    loading_array = np.empty((), dtype=dtype)
    read_from_dataset(dataset, selection, loading_array)
    return cast(_DT, loading_array[()])


def write_to_dataset(dataset: Dataset[_DT] | DatasetWrapper[_DT],
                     values: npt.NDArray[_DT],
                     selection: Selection) -> None:
    selection_shape = selection.compute_shape(dataset.shape)

    if values.size == np.prod(selection_shape) and values.shape != selection_shape:
        values = values.reshape(selection_shape)

    # we must make a copy here to ensure 'C' order
    values = values.copy()

    for dataset_idx, array_idx in selection.iter_h5(values.shape):
        dataset.write_direct(values, source_sel=array_idx, dest_sel=dataset_idx)
