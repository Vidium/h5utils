from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, SupportsIndex

import numpy as np
import numpy.typing as npt

import ch5mpy.indexing as ci


class Indexer(ABC):
    # region magic methods
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        pass

    # endregion

    # region attributes
    @property
    @abstractmethod
    def ndim(self) -> int:
        pass

    @property
    @abstractmethod
    def is_whole_axis(self) -> bool:
        pass

    # endregion

    # region methods
    @abstractmethod
    def as_numpy_index(self) -> npt.NDArray[np.int_] | slice | int | None:
        pass

    # endregion


def boolean_array_as_indexer(
    mask: npt.NDArray[np.bool_], shape: tuple[int, ...]
) -> tuple[ci.FullSlice | ci.ListIndex, ...]:
    assert mask.shape == shape

    if mask.ndim == 1 and np.all(mask):
        assert len(mask) == shape[0]
        return (ci.FullSlice.whole_axis(len(mask)),)

    return tuple(ci.ListIndex(e, max=s) for e, s in zip(np.where(mask), shape))


def as_indexer(obj: SupportsIndex | npt.NDArray[np.int_] | list[int] | slice | None, max: int) -> Indexer:
    if obj is None:
        return ci.NewAxis

    if isinstance(obj, (slice, range)):
        start = 0 if obj.start is None else obj.start
        step = 1 if obj.step is None else obj.step
        stop = max if obj.stop is None else obj.stop

        # /!\ np.sign(0) is 0 --> so when signs are different -- excluding 0 -- their product is -1
        if np.sign(start) * np.sign(step) == -1:
            return ci.ListIndex(np.arange(start, stop, step), max=max)

        if stop == start:
            return ci.EmptyList(max=max)

        return ci.FullSlice(start, stop, step, max=max)

    if isinstance(obj, (np.ndarray, list)):
        obj = np.array(obj)

        if obj.dtype == np.bool_:
            raise ValueError("Cannot convert boolean array, please use `boolean_array_as_indexer()` instead.")

        if obj.ndim == 0:
            return ci.SingleIndex(int(obj), max=max)

        if obj.size == 0:
            return ci.EmptyList(shape=obj.shape, max=max)

        return ci.ListIndex(obj, max=max)

    if isinstance(obj, SupportsIndex):
        return ci.SingleIndex(int(obj), max=max)

    raise TypeError(f"Cannot convert {obj} to indexer.")
