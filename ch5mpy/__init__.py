try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata  # type: ignore[no-redef, import-not-found]

import ch5mpy.functions.random
import ch5mpy.dict
from ch5mpy.array.array import H5Array
from ch5mpy.attributes import AttributeManager
from ch5mpy.dict import H5Dict
from ch5mpy.functions.creation_routines import empty, full, ones, zeros
from ch5mpy.list import H5List
from ch5mpy.names import H5Mode
from ch5mpy.np import arange_nd
from ch5mpy.objects import Dataset, File, Group
from ch5mpy.options import options, set_options
from ch5mpy.io import (
    read_object,
    write_dataset,
    write_datasets,
    write_object,
    write_objects,
)

from ch5mpy import indexing

random = ch5mpy.functions.random

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
    "options",
    "set_options",
    "indexing",
]

__version__ = metadata.version("ch5mpy")
