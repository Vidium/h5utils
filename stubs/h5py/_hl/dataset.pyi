from .base import HLObject
from ..h5d import DatasetID

class Dataset(HLObject):
    def __init__(self, bind: DatasetID, *, readonly: bool = False): ...
    @property
    def id(self) -> DatasetID: ...
