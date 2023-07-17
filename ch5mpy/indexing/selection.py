# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from itertools import chain, dropwhile, repeat, takewhile
from typing import Any, Generator, Generic, Iterable, Iterator, SupportsIndex, TypeVar, overload

import numpy as np
import numpy.typing as npt

from ch5mpy._typing import SELECTOR
from ch5mpy.indexing.list import ListIndex
from ch5mpy.indexing.optimization import get_valid_indices
from ch5mpy.indexing.slice import FullSlice
from ch5mpy.indexing.special import PLACEHOLDER, NewAxis, NewAxisType
from ch5mpy.indexing.typing import SELECTION_ELEMENT
from ch5mpy.indexing.utils import takewhile_inclusive
from ch5mpy.utils import is_sequence

# ====================================================
# code
_T = TypeVar("_T")


def boolean_mask_as_selection_elements(mask: npt.NDArray[np.bool_]) -> list[FullSlice | ListIndex]:
    if mask.ndim == 1 and np.all(mask):
        return [FullSlice.whole_axis(len(mask))]

    return [ListIndex(e) for e in np.where(mask)]


def get_indexer(
    obj: ListIndex | FullSlice | NewAxisType, sorted: bool = False, enforce_1d: bool = False, for_h5: bool = False
) -> int | npt.NDArray[np.int_] | slice:
    if isinstance(obj, NewAxisType):
        raise NotImplementedError

    if isinstance(obj, FullSlice):
        return obj.as_slice()

    if obj.size != 1:
        return obj.as_array(sorted=sorted, flattened=enforce_1d)

    if enforce_1d:
        return obj.squeeze().as_array().reshape((1,))

    if for_h5:
        return int(obj.squeeze().as_array()[()])

    return obj.squeeze().as_array()


def _within_bounds(
    obj: ListIndex | FullSlice, shape: tuple[int, ...], shape_index: int
) -> tuple[ListIndex | FullSlice]:
    max_ = shape[shape_index]

    if (isinstance(obj, FullSlice) and (obj.start < -max_ or obj.true_stop > max_)) or (
        isinstance(obj, ListIndex) and obj.size and (obj.min < -max_ or obj.max > max_)
    ):
        raise IndexError(f"Selection {obj} is out of bounds for axis {shape_index} with size {max_}.")

    return (obj,)


def _compute_shape_empty_dset(
    indices: tuple[SELECTION_ELEMENT, ...], arr_shape: tuple[int, ...], new_axes: bool
) -> tuple[int, ...]:
    if new_axes and any(
        dropwhile(lambda x: x, reversed(list(dropwhile(lambda x: x, [x is NewAxis for x in indices]))))
    ):
        raise NotImplementedError("Cannot have new axis besides first and last indices yet.")

    shape: tuple[int, ...] = ()

    for index in indices:
        if index is NewAxis:
            if not new_axes:
                continue

            shape += (1,)
            continue

        if index.ndim > 0:
            shape += (len(index),)

    return shape + arr_shape[len(indices) :]


def _get_sorting_indices(i: Any) -> npt.NDArray[np.int_] | slice:
    if isinstance(i, np.ndarray):
        return np.unique(i.flatten(), return_inverse=True)[1]  # type: ignore[no-any-return]
    return slice(None)


def _get_selection_at_index(
    sel_element: ListIndex, index: tuple[SELECTION_ELEMENT, ...], shape: tuple[int, ...]
) -> ListIndex:
    return sel_element.broadcast_to(shape)[index]


class Selection:
    __slots__ = "_indices", "_array_shape"

    # region magic methods
    def __init__(self, indices: Iterable[SELECTION_ELEMENT] | None, shape: tuple[int, ...], optimize: bool = True):
        if indices is None:
            self._indices: tuple[SELECTION_ELEMENT, ...] = ()
            return

        indices = tuple(indices)

        if len([i for i in indices if i is not NewAxis]) > len(shape):
            raise IndexError(f"Got too many indices ({len(indices)}) for shape {shape}.")

        it_indices = iter(indices)
        *_indices, first_list = takewhile_inclusive(lambda i: not isinstance(i, ListIndex), it_indices)

        largest_dim = 0 if first_list is None else max(i.ndim for i in indices if isinstance(i, ListIndex))
        self._indices = get_valid_indices(
            tuple(chain(_indices, () if first_list is None else (first_list.expand_to_dim(largest_dim),), it_indices)),
            shape=shape,
            optimize=optimize,
        )

        self._array_shape = shape

    def __repr__(self) -> str:
        return f"Selection{self._indices}\n" f"         {self.in_shape} --> {self.out_shape}"

    @overload
    def __getitem__(self, item: int) -> SELECTION_ELEMENT:
        ...

    @overload
    def __getitem__(self, item: slice | Iterable[int]) -> Selection:
        ...

    def __getitem__(self, item: int | slice | Iterable[int]) -> SELECTION_ELEMENT | Selection:
        if isinstance(item, int):
            if item >= len(self._indices):
                if item >= len(self.out_shape):
                    raise IndexError(f"Index {item} is out of range for {self.out_shape} selection.")

                return FullSlice.whole_axis(self.out_shape[item])

            return self._indices[item]

        elif isinstance(item, slice):
            return Selection(self._indices[item], shape=self._array_shape)

        return Selection([e for i, e in enumerate(self._indices) if i in item], shape=self._array_shape)

    def __matmul__(self, item: slice | Iterable[int]) -> Selection:
        """Same as __getitem__ but without optimization"""
        if isinstance(item, slice):
            return Selection(self._indices[item], shape=self._array_shape, optimize=False)

        return Selection([e for i, e in enumerate(self._indices) if i in item], shape=self._array_shape, optimize=False)

    def __iter__(self) -> Iterator[SELECTION_ELEMENT]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Selection):
            return False

        return (self._array_shape, self._indices) == (other._array_shape, other._indices)

    # endregion

    # region attributes
    @property
    def is_empty(self) -> bool:
        return len(self._indices) == 0

    @property
    def is_newaxis(self) -> npt.NDArray[np.bool_]:
        return np.array([x is NewAxis for x in self._indices])

    @property
    def is_list(self) -> npt.NDArray[np.bool_]:
        return np.array([isinstance(x, ListIndex) for x in self._indices])

    @property
    def is_slice(self) -> npt.NDArray[np.bool_]:
        return np.array([isinstance(x, FullSlice) for x in self._indices])

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._array_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        if np.prod(self._array_shape) == 0:
            return _compute_shape_empty_dset(self._indices, self._array_shape, new_axes=True)

        len_end_shape = len(self._indices) - sum(self.is_newaxis)
        return self._min_shape(new_axes=True) + self._array_shape[len_end_shape:]

    @property
    def out_shape_squeezed(self) -> tuple[int, ...]:
        if np.prod(self.in_shape) == 0:
            shape = _compute_shape_empty_dset(self._indices, self._array_shape, new_axes=False)

        else:
            len_end_shape = len(self._indices) - sum(self.is_newaxis)
            shape = self._min_shape(new_axes=False) + self._array_shape[len_end_shape:]

        return tuple(s for s, is_list in zip(shape, chain(self.is_list, repeat(False))) if not is_list or s != 1)

    # endregion

    # region methods
    @classmethod
    def from_selector(cls, index: SELECTOR | tuple[SELECTOR, ...], shape: tuple[int, ...]) -> Selection:
        if not isinstance(index, tuple):
            index = (index,)

        sel: tuple[SELECTION_ELEMENT, ...] = ()
        shape_index = 0

        for axis_index in index:
            if axis_index is None or axis_index is Ellipsis:
                sel += (NewAxis,)

            elif isinstance(axis_index, (slice, range)):
                sel += _within_bounds(
                    FullSlice(
                        axis_index.start,
                        axis_index.stop,
                        axis_index.step,
                        shape[shape_index],
                    ),
                    shape,
                    shape_index,
                )
                shape_index += 1

            elif is_sequence(axis_index) or isinstance(axis_index, SupportsIndex):
                axis_index = np.array(axis_index)

                if axis_index.dtype == bool:
                    for l_idx in boolean_mask_as_selection_elements(axis_index):
                        sel += _within_bounds(l_idx, shape, shape_index)

                else:
                    sel += _within_bounds(ListIndex(axis_index.astype(np.int64)), shape, shape_index)

                shape_index += 1

            else:
                raise ValueError(f"Invalid slicing object '{axis_index}'.")

        return Selection(sel, shape=shape)

    def get_indexers(
        self,
        sorted: bool = False,
        enforce_1d: bool = False,
        for_h5: bool = False,
    ) -> tuple[int | npt.NDArray[np.int_] | slice, ...]:
        if for_h5:
            return tuple(
                get_indexer(i, sorted=sorted, enforce_1d=enforce_1d, for_h5=True)
                for i in self._indices
                if not isinstance(i, NewAxisType)
            )

        return tuple(get_indexer(i, sorted=sorted, enforce_1d=enforce_1d, for_h5=False) for i in self._indices)

    def _min_shape(self, new_axes: bool) -> tuple[int, ...]:
        last_slice_position = np.argwhere(self.is_slice)[-1, 0] if any(self.is_slice) else np.inf
        first_list_position = np.argmax(self.is_list) if any(self.is_list) else -1

        slice_indices_shape = tuple(len(s) for s in self._indices if isinstance(s, FullSlice))
        list_indices_shape = np.broadcast_shapes(
            *(lst.shape for lst in self._indices if isinstance(lst, ListIndex) and lst.ndim > 0)
        )

        if first_list_position < last_slice_position:
            min_shape = list_indices_shape + slice_indices_shape

        else:
            min_shape = slice_indices_shape + list_indices_shape

        if not new_axes:
            return min_shape

        if any(dropwhile(lambda x: x, reversed(list(dropwhile(lambda x: x, self.is_newaxis))))):
            raise NotImplementedError("Cannot have new axis besides first and last indices yet.")

        new_axes_before = tuple(1 for _ in takewhile(lambda x: x, self.is_newaxis))
        new_axes_after = tuple(1 for _ in takewhile(lambda x: x, reversed(self.is_newaxis[len(new_axes_before) :])))

        return new_axes_before + min_shape + new_axes_after

    def cast_on(self, previous_selection: Selection) -> Selection:
        if self.in_shape != previous_selection.out_shape:
            raise ValueError(f"Cannot cast {self._array_shape} selection on {previous_selection.out_shape} selection.")

        if self.is_empty:
            return previous_selection

        if previous_selection.is_empty:
            return self

        casted_selection: list[SELECTION_ELEMENT | object] = [PLACEHOLDER for _ in enumerate(previous_selection)]
        self_indices = Queue(self._indices)
        previous_indices_pos = Queue(range(len(previous_selection)))

        while not self_indices.is_empty:
            index = self_indices.pop()

            if previous_indices_pos.is_empty:
                casted_selection.append(index)
                continue

            current_work_index = previous_indices_pos.pop()
            current_sel = previous_selection @ slice(current_work_index, None)
            nb_list_1d_plus = len([lst for lst in current_sel if isinstance(lst, ListIndex) and lst.ndim > 0])

            for sel_element in current_sel:
                if isinstance(sel_element, (FullSlice, NewAxisType)):
                    casted_selection[current_work_index] = sel_element[index]
                    break

                elif sel_element.shape == ():
                    casted_selection[current_work_index] = sel_element

                    if not previous_indices_pos.is_empty:
                        current_work_index = previous_indices_pos.pop()

                    else:
                        casted_selection.append(index)

                elif sel_element.shape == (1,) and len(current_sel._min_shape(new_axes=True)) < nb_list_1d_plus:
                    casted_selection[current_work_index] = sel_element[0]

                    if previous_indices_pos.is_empty:
                        break

                    current_work_index = previous_indices_pos.pop()

                else:
                    full_index = (index,) + self_indices.pop_at_most(sel_element.ndim - 1)
                    list_shape = next(list_shapes)

                    casted_selection[current_work_index] = sel_element.reshape(list_shape)[full_index]

                    if sel_element.ndim > 0:
                        # propagate selection to other 1D+ list indices
                        for prop_index, prop_sel in enumerate(
                            previous_selection @ slice(current_work_index + 1, None), start=current_work_index + 1
                        ):
                            if isinstance(prop_sel, ListIndex) and prop_sel.ndim > 0:
                                casted_selection[prop_index] = prop_sel.reshape(list_shape)[full_index]
                                previous_indices_pos.remove_at(prop_index)

                    break

        for i in previous_indices_pos.indices_not_popped:
            casted_selection[i] = previous_selection[i]

        if any(e is PLACEHOLDER for e in casted_selection):
            raise RuntimeError

        sel = Selection(
            [s for s in casted_selection if s is not None], shape=previous_selection.in_shape  # type: ignore[misc]
        )

        if (sel.in_shape, sel.out_shape) != (previous_selection.in_shape, self.out_shape):
            raise RuntimeError("Selection casting failed: could not maintain shape consistency.")

        return sel

    def iter_indexers(
        self,
    ) -> Generator[
        tuple[
            tuple[int | npt.NDArray[np.int_] | slice, ...],
            tuple[int | npt.NDArray[np.int_] | slice, ...],
        ],
        None,
        tuple[npt.NDArray[np.int_] | slice, ...],
    ]:
        # find the index of the first ListIndex in this selection
        list_indices = [i for i, x in enumerate(self._indices) if isinstance(x, ListIndex) and x.size > 1]

        # short indexing ------------------------------------------------------
        if len(list_indices) == 0 or len(list_indices) == 1 and self._indices[list_indices[0]].ndim == 1:
            # make sure there are at least 2 ListIndex, over-wise the selection can simply be returned as is
            dataset_sel = self.get_indexers(sorted=True, for_h5=True)
            loading_sel = tuple(0 for _ in takewhile(lambda x: x == 1, self._array_shape))

            yield dataset_sel, loading_sel

            sel_it = chain(self.get_indexers(for_h5=True), repeat(slice(None)))
            return tuple(slice(None) if s == 1 else _get_sorting_indices(next(sel_it)) for s in self.out_shape_squeezed)

        elif len(list_indices) == 1:
            for e_index, list_e in enumerate(np.array(self._indices[list_indices[0]]).flatten()):
                dataset_sel = tuple(
                    list_e if i == list_indices[0] else get_indexer(e, for_h5=True) for i, e in enumerate(self._indices)
                )
                loading_sel = tuple(0 for _ in takewhile(lambda x: x == 1, self._array_shape)) + (e_index,)

                yield dataset_sel, loading_sel

            sel_it = chain(self.get_indexers(for_h5=True), repeat(slice(None)))
            return tuple(slice(None) if s == 1 else _get_sorting_indices(next(sel_it)) for s in self.out_shape_squeezed)

        # long indexing -------------------------------------------------------
        # delete index of last list since we can subset with (at most) one list
        # last_list_index = list_indices.pop(-1)

        indices_dataset = [
            a.flatten()
            for a in np.broadcast_arrays(
                *(
                    idx
                    for i, idx in enumerate(self._indices)
                    if isinstance(idx, ListIndex) and i in list_indices and idx.size > 1
                )
            )
        ]
        indices_loading = zip(*[i.flatten() for i in np.indices(self._min_shape(new_axes=False))])

        for index_dataset, index_loading in zip(map(iter, zip(*indices_dataset)), indices_loading):
            dataset_sel = tuple(
                next(index_dataset) if i in list_indices else get_indexer(e, for_h5=True)
                for i, e in enumerate(self)
                if not isinstance(e, NewAxisType)
            )

            yield dataset_sel, index_loading

        return (slice(None),)

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
            raise RuntimeError("Empty queue")

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
            raise ValueError(f"Index {index} was already popped.")

        self._popped.append(index)

    # endregion
