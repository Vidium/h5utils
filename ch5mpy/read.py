# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import pickle
from typing import Any, cast

import numpy as np

import ch5mpy.dict
from ch5mpy.h5array.array import H5Array
from ch5mpy.objects.dataset import Dataset
from ch5mpy.objects.group import Group


# ====================================================
# code
def read_object(data: Dataset[Any] | Group) -> Any:
    """Read an object from a .h5 file"""
    if not isinstance(data, (Dataset, Group)):
        raise ValueError(f"Cannot read object from '{type(data)}'.")

    if isinstance(data, Group):
        h5_type = data.attrs.get("__h5_type__", "<UNKNOWN>")
        if h5_type != "object":
            return ch5mpy.dict.H5Dict(data)

        h5_class = data.attrs.get("__h5_class__", None)
        if h5_class is None:
            raise ValueError("Cannot read object with unknown class.")

        data_class = pickle.loads(h5_class)

        if not hasattr(data_class, "__h5_read__"):
            raise ValueError(
                f"Don't know how to read {data_class} since it does not " f"implement the '__h5_read__' method."
            )

        return data_class.__h5_read__(data)

    if data.ndim == 0:
        if np.issubdtype(data.dtype, np.void):
            return pickle.loads(data[()])  # type: ignore[arg-type]

        if data.dtype == object:
            return cast(bytes, data[()]).decode()

        return data[()]

    return H5Array(data)
