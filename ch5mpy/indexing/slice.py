from __future__ import annotations

from math import lcm
from typing import Any, Iterable, Literal

import numpy as np
import numpy.typing as npt

from ch5mpy.indexing.base import Indexer, as_indexer
from ch5mpy.indexing.single import SingleIndex
from ch5mpy.indexing.special import NewAxis


def _positive(value: int, max: int) -> int:
    return value if value >= 0 else max + value


class FullSlice(Indexer):
    # region magic methods
    def __init__(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
        max: int,
    ):
        if step == 0:
            raise ValueError("FullSlice step cannot be zero.")

        start = 0 if start is None else start
        stop = max if stop is None else stop

        self._start = _positive(0 if start is None else start, max)
        self._step = 1 if step is None else abs(step)
        self._stop = _positive(min(stop, max), max)
        self._max = max

        if self._max < self._start and self._max < self._stop:
            raise IndexError(f"Selection {slice(start, stop, step)} is out of bounds for axis with size {max}.")

        if len(self) == 0:
            raise ValueError("FullSlice cannot have length 0.")

    def __repr__(self) -> str:
        if self.is_one_element:
            return f"{type(self).__name__}(<{self.start}> | {self._max})"

        if self.is_whole_axis:
            return f"{type(self).__name__}(* | {self._max})"

        return f"{type(self).__name__}({self._start}, {self._stop}, {self._step} | {self._max})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FullSlice):
            raise NotImplementedError

        if (
            self._start == other.start
            and self._stop == other.stop
            and self._step == other.step
            and self._max == other.max
        ):
            return True

        return False

    def __len__(self) -> int:
        if self._stop == self._start:
            return 0

        return (self.true_stop - self._start) // self._step + 1

    def __getitem__(self, item: Indexer) -> Indexer:
        if isinstance(item, FullSlice):
            return FullSlice(
                start=self._start + item.start * self._step,
                stop=self._start + item.stop * self._step,
                step=lcm(self._step, item.step),
                max=self._max,
            )

        if item is NewAxis:
            raise RuntimeError

        return as_indexer(np.array(self)[item.as_numpy_index()], max=self._max)

    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        return np.array(range(self.start, self._stop, self._step), dtype=dtype)

    # endregion

    # region attributes
    @property
    def start(self) -> int:
        return self._start

    @property
    def stop(self) -> int:
        return self._stop

    @property
    def true_stop(self) -> int:
        """Get the true last int in this slice, if converted to a list."""
        if self._start == self._stop:
            return self._stop

        return self._start + (self._stop - 1 - self._start) // self._step * self._step

    @property
    def step(self) -> int:
        return self._step

    @property
    def max(self) -> int:
        return self._max

    @property
    def ndim(self) -> Literal[1]:
        return 1

    @property
    def shape(self) -> tuple[int, ...]:
        return (len(self),)

    # endregion

    # region predicates
    @property
    def is_whole_axis(self) -> bool:
        return self._start == 0 and self._step == 1 and self._stop == self._max

    @property
    def is_one_element(self) -> bool:
        return len(self) == 1

    # endregion

    # region methods
    @classmethod
    def whole_axis(cls, max: int) -> FullSlice:
        return FullSlice(0, max, 1, max)

    @classmethod
    def one(cls, element: int, max: int) -> FullSlice:
        return FullSlice(element, element + 1, 1, max)

    @classmethod
    def from_slice(cls, s: slice | range, max: int) -> FullSlice:
        return FullSlice(s.start, s.stop, s.step, max)

    def as_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    def as_numpy_index(self) -> slice:
        return self.as_slice()

    def shift_to_zero(self) -> FullSlice:
        return FullSlice(0, self.stop - self.start, self.step, self._max)

    # endregion


def map_slice(index: Iterable[FullSlice | SingleIndex], shift_to_zero: bool = False) -> tuple[slice, ...]:
    if shift_to_zero:
        return tuple(fs.shift_to_zero().as_slice() for fs in index if isinstance(fs, FullSlice))

    else:
        return tuple(fs.as_slice() for fs in index)
