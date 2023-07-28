from __future__ import annotations

import pickle
from typing import Any, Literal, cast

import numpy as np

import ch5mpy.dict
from ch5mpy.h5array.array import H5Array
from ch5mpy.objects import Dataset, Group


def _handle_read_error(data: Group, error: Literal["ignore", "raise"], msg: str) -> ch5mpy.dict.H5Dict[Any]:
    if error == "raise":
        raise ValueError(msg)

    else:
        return ch5mpy.dict.H5Dict(data, annotation=f"Failed reading object: {msg}")


def read_object(
    data: Dataset[Any] | Group, error: Literal["ignore", "raise"] = "raise", read_object: bool = True
) -> Any:
    """Read an object from a .h5 file"""
    if not isinstance(data, (Dataset, Group)):
        raise ValueError(f"Cannot read object from '{type(data)}'.")

    if isinstance(data, Group):
        h5_type = data.attrs.get("__h5_type__", "<UNKNOWN>")
        if not read_object or h5_type != "object":
            return ch5mpy.dict.H5Dict(data)

        h5_class = data.attrs.get("__h5_class__", None)
        if h5_class is None:
            return _handle_read_error(data, error, "Cannot read object with unknown class.")

        data_class = pickle.loads(h5_class)
        if not hasattr(data_class, "__h5_read__"):
            return _handle_read_error(
                data,
                error,
                f"Don't know how to read {data_class} since it does not implement " f"the '__h5_read__' method.",
            )

        try:
            return data_class.__h5_read__(ch5mpy.dict.H5Dict(data))

        except Exception as e:
            return _handle_read_error(data, error, str(e))

    if data.ndim == 0:
        if np.issubdtype(data.dtype, np.void):
            return pickle.loads(data[()])  # type: ignore[arg-type]

        if data.dtype == object:
            return cast(bytes, data[()]).decode()

        return data[()]

    return H5Array(data)
