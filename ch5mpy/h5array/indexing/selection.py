# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from itertools import islice

import numpy.typing as npt
from typing import Any
from typing import TypeVar
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import overload
from typing import Generator

from ch5mpy._typing import SELECTOR
from ch5mpy.h5array.indexing.list import ListIndex
from ch5mpy.h5array.indexing.slice import FullSlice
from ch5mpy.utils import is_sequence


# ====================================================
# code
_T = TypeVar('_T')


def _cast_h5(obj: ListIndex | FullSlice) -> int | npt.NDArray[np.int_] | slice:
    if isinstance(obj, FullSlice):
        return obj.as_slice()

    if obj.size > 1:
        return obj.as_array()

    return int(obj.squeeze().as_array()[()])


def _gets_whole_dataset(index: SELECTOR | tuple[SELECTOR, ...]) -> bool:
    return (isinstance(index, tuple) and index == ()) or \
        (isinstance(index, slice) and index == slice(None)) or \
        (is_sequence(index) and all((e is True for e in index)))


def _within_bounds(obj: ListIndex | FullSlice, shape: tuple[int, ...], shape_index: int) \
        -> tuple[ListIndex | FullSlice]:
    max_ = shape[shape_index]

    if (isinstance(obj, FullSlice) and (obj.start < -max_ or obj.true_stop > max_)) or \
            (isinstance(obj, ListIndex) and (obj.min < -max_ or obj.max > max_)):
        raise IndexError(f"Selection {obj} is out of bounds for axis {shape_index} with size {max_}.")

    return obj,


class Selection:

    # region magic methods
    def __init__(self,
                 indices: Iterable[ListIndex | FullSlice] | None = None):
        if indices is None:
            self._indices: tuple[ListIndex | FullSlice, ...] = ()

        else:
            indices = tuple(indices)
            parsed_indices: tuple[ListIndex | FullSlice, ...] = ()
            first_list = True

            for i in indices:
                if isinstance(i, ListIndex):
                    if first_list:
                        largest_dim = max(indices, key=lambda x: x.ndim if isinstance(x, ListIndex) else 0).ndim
                        parsed_indices += (i.expand(largest_dim),)
                        first_list = False

                    else:
                        parsed_indices += (i.squeeze(),)

                else:
                    parsed_indices += (i,)

            self._indices = parsed_indices

    def __repr__(self) -> str:
        return f"Selection{self._indices}"

    @overload
    def __getitem__(self, item: int) -> ListIndex | FullSlice: ...
    @overload
    def __getitem__(self, item: slice | Iterable[int]) -> Selection: ...
    def __getitem__(self, item: int | slice | Iterable[int]) -> ListIndex | FullSlice | Selection:
        if isinstance(item, int):
            return self._indices[item]

        elif isinstance(item, slice):
            return Selection(self._indices[item])

        else:
            return Selection([e for i, e in enumerate(self._indices) if i in item])

    def __iter__(self) -> Iterator[ListIndex | FullSlice]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Selection):
            return False

        return self._indices == other._indices

    # endregion

    # region attributes
    @property
    def is_empty(self) -> bool:
        return len(self._indices) == 0

    @property
    def min_shape(self) -> tuple[int, ...]:
        is_slice = list(map(lambda x: isinstance(x, FullSlice), self._indices))
        last_slice_position = np.argwhere(is_slice)[-1, 0] if any(is_slice) else np.inf
        is_list = list(map(lambda x: isinstance(x, ListIndex), self._indices))
        first_list_position = np.argmax(is_list) if any(is_list) else -1

        slice_indices_shape = tuple(len(s) for s in self._indices if isinstance(s, FullSlice))
        list_indices_shape = np.broadcast_shapes(*(lst.shape for lst in self._indices if isinstance(lst, ListIndex)))

        if first_list_position < last_slice_position:
            return list_indices_shape + slice_indices_shape

        else:
            return slice_indices_shape + list_indices_shape

    # endregion

    # region methods
    @classmethod
    def from_selector(cls,
                      index: SELECTOR | tuple[SELECTOR, ...],
                      shape: tuple[int, ...]) -> Selection:
        if _gets_whole_dataset(index):
            return Selection()

        if not isinstance(index, tuple):
            index = (index,)

        sel: tuple[ListIndex | FullSlice, ...] = ()
        shape_index = 0

        for i, axis_index in enumerate(index):
            if isinstance(axis_index, (slice, range)):
                sel += _within_bounds(
                    FullSlice(axis_index.start, axis_index.stop, axis_index.step, shape[shape_index]),
                    shape, shape_index
                )
                shape_index += 1

            elif is_sequence(axis_index) or isinstance(axis_index, int):
                axis_index = np.array(axis_index)

                if axis_index.dtype == bool:
                    for e in np.where(axis_index):
                        sel += _within_bounds(ListIndex(e), shape, shape_index)
                        shape_index += 1

                else:
                    sel += _within_bounds(ListIndex(axis_index), shape, shape_index)
                    shape_index += 1

            else:
                raise ValueError(f"Invalid slicing object '{axis_index}'.")

        return Selection(sel)

    def get(self) -> tuple[int | npt.NDArray[np.int_] | slice, ...]:
        return tuple(_cast_h5(i) for i in self._indices)

    def compute_shape(self, arr_shape: tuple[int, ...]) -> tuple[int, ...]:
        return self.min_shape + arr_shape[len(self._indices):]

    def cast_on(self, selection: Selection) -> Selection:
        casted_selection: list[ListIndex | FullSlice | None] = [None for _ in enumerate(selection)]
        self_indices = Queue(self._indices)
        other_indices = Queue(list(range(len(selection))))

        while not self_indices.is_empty:
            index = self_indices.pop()

            if other_indices.is_empty:
                casted_selection.append(index)
                continue

            current_work_index = other_indices.pop()
            current_sel = selection[current_work_index:]
            nb_list_1d_plus = len([lst for lst in current_sel if isinstance(lst, ListIndex) and lst.ndim > 0])

            for sel_element in current_sel:
                if isinstance(sel_element, FullSlice):
                    casted_selection[current_work_index] = sel_element[index]
                    break

                else:
                    if sel_element.shape == () \
                            or sel_element.shape == (1,) and len(current_sel.min_shape) < nb_list_1d_plus:
                        # if only one element of the axis is selected, skip this sel element from the casting
                        casted_selection[current_work_index] = sel_element.squeeze()

                        if other_indices.is_empty:
                            casted_selection.append(index)
                            break

                        current_work_index = other_indices.pop()

                    else:
                        full_index = (index,) + self_indices.pop_at_most(sel_element.ndim - 1)
                        casted_selection[current_work_index] = sel_element[full_index]

                        if sel_element.ndim == 1:
                            # propagate selection to other 1D+ list indices
                            for prop_index, prop_sel in enumerate(selection[current_work_index + 1:],
                                                                  start=current_work_index + 1):
                                if isinstance(prop_sel, ListIndex) and prop_sel.ndim == 1:
                                    casted_selection[prop_index] = prop_sel[index]
                                    other_indices.remove_at(prop_index)

                        break

        for i in other_indices.indices_not_popped:
            casted_selection[i] = selection[i]

        if any([s is None for s in casted_selection]):
            raise RuntimeError

        return Selection(casted_selection)                                                      # type: ignore[arg-type]

    def iter_h5(self, array_ndim: int) -> Generator[
        tuple[tuple[int | npt.NDArray[np.int_] | slice, ...],
              tuple[int | slice, ...]],
        None,
        None
    ]:
        # find the index of the first ListIndex in this selection
        list_indices = [i for i, x in enumerate(self._indices) if isinstance(x, ListIndex) and x.size > 1]

        if len(list_indices) == 0 or len(list_indices) == 1 and self._indices[list_indices[0]].ndim == 1:
            # make sure there are at least 2 ListIndex, over-wise the selection can simply be returned as is
            yield self.get(), ()
            return

        whole_lists = [a.flatten() for a in np.broadcast_arrays(*(self._indices[i] for i in list_indices))]

        # noinspection PyTypeChecker
        for index_array, index_set in enumerate(map(iter, zip(*whole_lists))):
            dataset_sel: tuple[int | npt.NDArray[np.int_] | slice, ...] = \
                tuple(next(index_set) if i in list_indices else _cast_h5(e)
                      for i, e in enumerate(self))

            loading_sel: tuple[int | slice, ...] = \
                tuple(index_array if i in list_indices else slice(None)
                      for i, e in enumerate(islice(self, 0, array_ndim)))

            yield dataset_sel, loading_sel

    # endregion


class Queue(Generic[_T]):

    # region magic methods
    def __init__(self, elements: Iterable[_T]):
        self._elements = tuple(elements)
        self._popped: list[int] = []

    def __repr__(self) -> str:
        return f"Queue([{', '.join(str(e) if i not in self._popped else '--' for i, e in enumerate(self._elements))}])"

    # endregion

    # region attributes
    @property
    def is_empty(self) -> bool:
        return self._next_index() is None

    @property
    def indices_not_popped(self) -> list[int]:
        return [i for i, _ in enumerate(self._elements) if i not in self._popped]

    # endregion

    # region methods
    def _next_index(self) -> int | None:
        for i, _ in enumerate(self._elements):
            if i not in self._popped:
                return i

        return None

    def pop(self) -> _T:
        i = self._next_index()
        if i is None:
            raise RuntimeError('Empty queue')

        self._popped.append(i)
        return self._elements[i]

    def pop_at_most(self, n: int) -> tuple[_T, ...]:
        popped_elements: tuple[_T, ...] = ()
        for _ in range(n):
            try:
                popped_elements += (self.pop(),)
            except RuntimeError:
                break

        return popped_elements

    def remove_at(self, index: int) -> None:
        if index in self._popped:
            raise ValueError(f'Index {index} was already popped.')

        self._popped.append(index)

    # endregion
