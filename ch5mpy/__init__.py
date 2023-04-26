# coding: utf-8
# Created on 13/12/2022 14:41
# Author : matteo

# ====================================================
# imports
from .attributes import AttributeManager
from .dict import H5Dict
from .h5array.array import H5Array
from .h5array.creation_routines import empty, full, ones, zeros
from .list import H5List
from .names import H5Mode
from .np import arange_nd
from .objects.dataset import Dataset
from .objects.group import File, Group
from .read import read_object
from .write import (
    write_dataset,
    write_datasets,
    write_object,
    write_objects,
)

# ====================================================
# code
__all__ = [
    "File",
    "Group",
    "Dataset",
    "H5Dict",
    "H5List",
    "H5Array",
    "AttributeManager",
    "write_dataset",
    "write_datasets",
    "write_object",
    "write_objects",
    "read_object",
    "H5Mode",
    "arange_nd",
    "empty",
    "zeros",
    "ones",
    "full",
]

__version__ = "0.1.3"
