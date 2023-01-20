# coding: utf-8

# ====================================================
# imports
import pickle
import numpy as np
from h5py import string_dtype
from numbers import Number

from typing import Any
from typing import Mapping
from typing import Collection
from typing import TypeGuard

from h5utils.pickle.wrap import File
from h5utils.pickle.wrap import Group


# ====================================================
# code
def write_attribute(group: Group, name: str, obj: Any) -> None:
    """Write a simple object as a H5 group attribute."""
    try:
        group.attrs[name] = "None" if obj is None else obj

    except TypeError:
        # since <obj> cannot be stored directly, we pickle it
        # here we use numpy's void type to allow storing bytes generated by pickle
        group.attrs[name] = np.void(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))


def write_attributes(group: Group, **kwargs: Any) -> None:
    """Write multiple object as h5 group attributes."""
    for name, obj in kwargs.items():
        write_attribute(group, name, obj)


def write_dataset(loc: Group | File, name: str, obj: Any) -> None:
    """Write an array-like object to a H5 dataset."""
    if isinstance(obj, Mapping):
        group = loc.create_group(name)
        write_datasets(group, **obj)

    else:
        array = np.array(obj)

        if np.issubdtype(array.dtype, np.str_):
            loc.create_dataset(name, data=array.astype(object), dtype=string_dtype())

        else:
            loc.create_dataset(name, data=array)


def write_datasets(loc: Group | File, **kwargs: Any) -> None:
    """Write multiple array-like objects to H5 datasets."""
    for name, obj in kwargs.items():
        write_dataset(loc, name, obj)


def is_collection(obj: Any) -> TypeGuard[Collection[Any]]:
    """Is the object a sequence of objects ? (excluding strings and byte objects.)"""
    return isinstance(obj, Collection) and not isinstance(
        obj, (str, bytes, bytearray, memoryview)
    )


def write_object(loc: Group | File, name: str, obj: Any) -> None:
    """Write any object to a H5 file."""
    if isinstance(obj, Mapping):
        group = loc.create_group(name)
        write_objects(group, **obj)

    elif is_collection(obj) or isinstance(obj, (Number, str)):
        write_dataset(loc, name, obj)

    else:
        raise NotImplementedError


def write_objects(loc: Group | File, **kwargs: Any) -> None:
    """Write multiple objects of any type to a H5 file."""
    for name, obj in kwargs.items():
        write_object(loc, name, obj)
