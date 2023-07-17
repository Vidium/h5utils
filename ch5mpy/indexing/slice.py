# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from math import lcm
from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
import numpy.typing as npt

from ch5mpy.indexing.list import ListIndex

if TYPE_CHECKING:
    from ch5mpy.indexing.typing import SELECTION_ELEMENT


# ====================================================
# code
def _positive(value: int, max_: int) -> int:
    return value if value >= 0 else max_ + value


class FullSlice:
    # region magic methods
    def __new__(cls, start: int | None, stop: int | None, step: int | None, max_: int):
        if start is not None and step is not None and np.sign(start) * np.sign(step) == -1:
            # /!\ np.sign(0) is 0 --> so when signs are different -- excluding 0 -- their product is -1
            return ListIndex(np.arange(start, max_ if stop is None else stop, step))

        return super().__new__(cls)

    def __init__(
        self,
        start: int | None,
        stop: int | None,
        step: int | None,
        max_: int,
    ):
        if step == 0:
            raise ValueError("FullSlice step cannot be zero.")

        start = 0 if start is None else start
        stop = max_ if stop is None else stop

        self._start = _positive(0 if start is None else start, max_)
        self._step = 1 if step is None else step
        self._stop = _positive(min(stop, max_), max_)
        self._max = max_

        assert self._max >= self._start and self._max >= self._stop

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

    def __getitem__(self, item: SELECTION_ELEMENT) -> ListIndex | FullSlice:
        from ch5mpy.indexing.special import NewAxisType

        if isinstance(item, FullSlice):
            return FullSlice(
                start=self._start + item.start * self._step,
                stop=self._start + item.stop * self._step,
                step=lcm(self._step, item.step),
                max_=self._max,
            )

        if isinstance(item, NewAxisType):
            raise RuntimeError

        return ListIndex(np.array(self)[item])

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

        last = np.arange(self._stop - self._step, self._stop, dtype=int)
        return int(last[last % self._step == self._start % self._step][-1])

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
    def whole_axis(cls, max_: int) -> FullSlice:
        return FullSlice(0, max_, 1, max_)

    @classmethod
    def one(cls, element: int, max_: int | None = None) -> FullSlice:
        max_ = element + 1 if max_ is None else max_
        return FullSlice(element, element + 1, 1, max_)

    @classmethod
    def from_slice(cls, s: slice | range, max_: int) -> FullSlice:
        return FullSlice(s.start, s.stop, s.step, max_)

    def as_slice(self) -> slice:
        return slice(self.start, self.stop, self.step)

    def shift_to_zero(self) -> FullSlice:
        return FullSlice(0, self.stop - self.start, self.step, self._max)

    # endregion


def map_slice(index: Iterable[FullSlice], shift_to_zero: bool = False) -> tuple[slice, ...]:
    if shift_to_zero:
        return tuple(fs.shift_to_zero().as_slice() for fs in index)

    else:
        return tuple(fs.as_slice() for fs in index)
