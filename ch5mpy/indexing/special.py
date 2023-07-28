from __future__ import annotations

from typing import Any, NoReturn

import numpy as np
import numpy.typing as npt

from ch5mpy.indexing.base import Indexer


class NewAxisType(Indexer):
    # region magic methods
    def __new__(cls) -> NewAxisType:
        return NewAxis

    def __repr__(self) -> str:
        return "<NewAxis>"

    def __len__(self) -> int:
        return 1

    def __eq__(self, other: Any) -> bool:
        return other is NewAxis

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        raise TypeError

    # endregion

    # region attributes
    @property
    def shape(self) -> tuple[int, ...]:
        return (1,)

    @property
    def ndim(self) -> int:
        return 1

    @property
    def is_whole_axis(bool) -> NoReturn:
        raise RuntimeError

    # endregion

    # region methods
    def as_numpy_index(self) -> None:
        return None

    # endregion


NewAxis = object.__new__(NewAxisType)


class EmptyList(Indexer):
    # region magic methods
    def __init__(self, *, max: int, shape: tuple[int, ...] = (0,)):
        assert np.prod(shape) == 0
        self._shape = shape
        self._max = max

    def __repr__(self) -> str:
        return f"<EmptyList {self.shape}>"

    def __len__(self) -> int:
        return 0

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, EmptyList):
            raise NotImplementedError

        return self._shape == other.shape

    def __getitem__(self, item: Indexer) -> EmptyList:
        arr = np.empty(self._shape)[item.as_numpy_index()]
        return EmptyList(shape=arr.shape, max=self._max)

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        return np.empty(self._shape, dtype=dtype)

    # endregion

    # region attributes
    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    @property
    def ndim(self) -> int:
        return len(self._shape)

    @property
    def is_whole_axis(self) -> bool:
        return self._max == 0

    # endregion

    # region methods
    def as_array(self) -> npt.NDArray[np.int_]:
        return self.__array__(dtype=np.int32)

    def as_numpy_index(self) -> npt.NDArray[np.int_]:
        return self.__array__(dtype=np.int32)

    # endregion


class PlaceHolderType:
    def __new__(self) -> PlaceHolderType:
        return PLACEHOLDER

    def __repr__(self) -> str:
        return "<placeholder>"


PLACEHOLDER = object.__new__(PlaceHolderType)
