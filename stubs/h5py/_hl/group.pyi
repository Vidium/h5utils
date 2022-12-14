from typing import Any
from typing import Iterator

import numpy as np

from .base import HLObject
from .base import MutableMappingHDF5
from . import dataset
from . import datatype
from . import attrs

class Group(
    HLObject, MutableMappingHDF5[str, Group | dataset.Dataset | datatype.Datatype]
):
    def __delitem__(self, name: str) -> None: ...
    def __getitem__(self, name: str) -> Group | dataset.Dataset | datatype.Datatype: ...
    def __setitem__(self, name: str, obj: Any) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    @property
    def attrs(self) -> attrs.AttributeManager: ...
