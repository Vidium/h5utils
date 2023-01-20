import numpy as np

from typing import Any
from typing import Iterator
from typing import Collection

from .base import HLObject
from .base import MutableMappingHDF5
from .dataset import Dataset
from .datatype import Datatype
from . import attrs
from ..h5g import GroupID

class Group(HLObject, MutableMappingHDF5[str, Group | Dataset | Datatype]):
    def __init__(self, bind: GroupID): ...
    def __delitem__(self, name: str) -> None: ...
    def __getitem__(
        self, name: str | slice | tuple[()]
    ) -> Group | Dataset | Datatype: ...
    def __setitem__(self, name: str, obj: Any) -> None: ...
    def __iter__(self) -> Iterator[str]: ...
    def __len__(self) -> int: ...
    @property
    def attrs(self) -> attrs.AttributeManager: ...
    def create_group(self, name: str, track_order: bool | None = None) -> Group: ...
    def create_dataset(
        self,
        name: str,
        shape: tuple[()] | tuple[int, ...] | None = None,
        dtype: str | np.dtype[Any] | None = None,
        data: Collection[Any] | None = None,
        **kwds: Any
    ) -> Dataset: ...
    @property
    def id(self) -> GroupID: ...
