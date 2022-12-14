# coding: utf-8
# Created on 13/12/2022 14:46
# Author : matteo

# ====================================================
# imports
from h5py import Group

from typing import Any


# ====================================================
# code
def write_attribute(group: Group,
                    name: str,
                    obj: Any) -> None:
    """Write a simple object as a H5 group attribute."""
    group.attrs[name] = 'None' if obj is None else obj


def write_attributes(group: Group,
                     **kwargs: Any) -> None:
    """Write multiple object as h5 group attributes."""
    for name, obj in kwargs.keys():
        write_attribute(group, name, obj)
