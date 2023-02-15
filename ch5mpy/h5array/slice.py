# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any
from typing import Iterable


# ====================================================
# code
class FullSlice:

    # region magic methods
    def __init__(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
        max_: int,
    ):
        self._start = start or 0
        self._step = step or 1
        self._stop = stop or max_
        self._max = max_

        if self._start < 0:
            self._start = self._stop + self._start

    def __repr__(self) -> str:
        if self.is_one_element():
            return f"{type(self).__name__}(<{self.start}> | {self._max})"

        if self.is_whole_axis():
            return f"{type(self).__name__}(* | {self._max})"

        return f"{type(self).__name__}({self._start}, {self._stop}, {self._step} | {self._max})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, FullSlice):
            raise NotImplementedError

        if self._start == other.start and self._stop == other.stop and self._step == other.step \
                and self._max == other.max:
            return True

        return False

    def __len__(self) -> int:
        return (self.true_stop - self._start) // self._step + 1

    @classmethod
    def whole_axis(cls, max_: int) -> FullSlice:
        return FullSlice(0, max_, 1, max_)

    @classmethod
    def one(cls,
            element: int,
            max_: int | None = None) -> FullSlice:
        max_ = element + 1 if max_ is None else max_
        return FullSlice(element, element+1, 1, max_)

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
        """Get the true last int in ths slice, if converted to a list."""
        if self._start == self._stop:
            return self._stop

        last = np.arange(self._stop - self._step, self._stop, dtype=int)
        return int(last[last % self._step == self._start % self._step][-1])

    @property
    def step(self) -> int:
        return self._step

    @property
    def max(self) -> int:
        return self._max

    # endregion

    # region predicates
    def is_whole_axis(self, max_: int | None = None) -> bool:
        max_ = max_ or self._max
        return self._start == 0 and self._step == 1 and self._stop == max_

    def is_one_element(self) -> bool:
        return len(self) == 1

    # endregion

    # region methods
    def as_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    def shift_to_zero(self) -> FullSlice:
        return FullSlice(0, self.stop - self.start, self.step, self._max)

    # endregion


def map_slice(index: Iterable[FullSlice]) -> tuple[slice, ...]:
    return tuple(fs.as_slice() for fs in index)
