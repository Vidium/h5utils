# coding: utf-8
# Created on 13/12/2022 14:46
# Author : matteo

# ====================================================
# imports
import pickle
import numpy as np
from h5py import File
from h5py import Group
from h5py import string_dtype

from typing import Any


# ====================================================
# code
def write_attribute(group: Group, name: str, obj: Any) -> None:
    """Write a simple object as a H5 group attribute."""
    try:
        group.attrs[name] = "None" if obj is None else obj

    except TypeError:
        group.attrs[name] = np.void(pickle.dumps(obj))


def write_attributes(group: Group, **kwargs: Any) -> None:
    """Write multiple object as h5 group attributes."""
    for name, obj in kwargs.items():
        write_attribute(group, name, obj)


def write_object(loc: Group | File, name: str, obj: Any) -> None:
    if isinstance(obj, dict):
        group = loc.create_group(name)
        write_objects(group, **obj)

    else:
        array = np.array(obj)

        if np.issubdtype(array.dtype, np.str_):
            loc.create_dataset(name, data=array.astype(object), dtype=string_dtype())

        else:
            loc.create_dataset(name, data=array)


def write_objects(loc: Group | File, **kwargs: Any) -> None:
    for name, obj in kwargs.items():
        write_object(loc, name, obj)
