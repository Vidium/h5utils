# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np

import numpy.typing as npt
from typing import Any


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
        return f"{type(self).__name__}({self._start}, {self._stop}, {self._step} | {self._max})"

    def __len__(self) -> int:
        return (self.true_stop - self._start) // self._step + 1

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

    # endregion

    # region methods
    def as_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    # endregion
