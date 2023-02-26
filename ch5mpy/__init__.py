# coding: utf-8
# Created on 13/12/2022 14:41
# Author : matteo

# ====================================================
# imports
from .objects.group import File
from .objects.group import Group
from .objects.dataset import Dataset
from .h5dict import H5Dict
from .h5array.array import H5Array
from .write import (
    write_attribute,
    write_attributes,
    write_dataset,
    write_datasets,
    write_object,
    write_objects,
)
from .names import H5Mode
from .np import *

# ====================================================
# code
__all__ = [
    "File",
    "Group",
    "Dataset",
    "H5Dict",
    "H5Array",
    "write_attribute",
    "write_attributes",
    "write_dataset",
    "write_datasets",
    "write_object",
    "write_objects",
    "H5Mode",
]
