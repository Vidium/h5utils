from itertools import chain, dropwhile

import numpy as np
import numpy.typing as npt

from ch5mpy.indexing.list import ListIndex
from ch5mpy.indexing.slice import FullSlice
from ch5mpy.indexing.special import NewAxis
from ch5mpy.indexing.typing import SELECTION_ELEMENT
from ch5mpy.indexing.utils import takewhile_inclusive

_SEL_IDS = tuple[SELECTION_ELEMENT, ...]
_SHAPE = tuple[int, ...]


def get_valid_indices(indices: _SEL_IDS, shape: _SHAPE, optimize: bool) -> _SEL_IDS:
    if not len(indices):
        return ()

    if optimize and _gets_whole_dataset(indices):
        return ()

    # common_shape = np.broadcast_shapes(*(i.shape for i in indices if isinstance(i, ListIndex)))
    # indices = tuple(i.broadcast_to(common_shape) if isinstance(i, ListIndex) else i for i in indices)

    optimized_indices = None

    if (
        optimize
        and any(isinstance(i, ListIndex) for i in indices)
        and all(isinstance(i, (FullSlice, ListIndex)) for i in indices)
    ):
        optimized_indices = _inverse_ix_for_n_lists(indices, shape)

    if optimized_indices is None:
        index_first_list = next((i for i, e in enumerate(indices) if isinstance(e, ListIndex)), 0)

        optimized_indices = tuple(
            i.squeeze() if isinstance(i, ListIndex) and idx > index_first_list else i for idx, i in enumerate(indices)
        )

    if optimize:
        if _gets_whole_dataset(optimized_indices):
            return ()

        optimized_indices = _drop_slice_on_whole_axis(optimized_indices)

    return optimized_indices


def _gets_whole_dataset(indices: _SEL_IDS) -> bool:
    return all(isinstance(i, FullSlice) and i.is_whole_axis for i in indices)


def _drop_slice_on_whole_axis(indices: _SEL_IDS) -> _SEL_IDS:
    return tuple(dropwhile(lambda i: isinstance(i, FullSlice) and i.is_whole_axis, indices[::-1]))[::-1]


def _as_slice(array: npt.NDArray[np.int_], max_: int) -> FullSlice:
    if len(array) == 0:
        return FullSlice(0, 0, 1, max_=max_)

    if len(array) == 1:
        return FullSlice(array[0], array[0] + 1, 1, max_=max_)

    step = array[1] - array[0]
    miss_steps = np.where(np.ediff1d(array) != step)[0]

    if len(miss_steps):
        raise ValueError

    array[array < 0] = max_ + array[array < 0]

    return FullSlice(array[0], array[-1] + step, step, max_=max_)


def _slices_inverse_ix_(indices: tuple[ListIndex, ...], shape: _SHAPE) -> _SEL_IDS | None:
    """
    From indexing arrays, return (if possible) the tuple of slices that would lead to the same indexing since indexing
    from slices leads to much fewer read operations.
    Also, return the number of NewAxes to add before and after the tuple of slices so as to maintain the resulting
    array's dimensions.

    This is effectively the inverse of the numpy.ix_ function, with indexing arrays converted to slices.
    """
    ndim = max(i.ndim for i in indices)

    if len(indices) > ndim:
        return None

    shapes = np.array([i.expand_to_dim(ndim).shape for i in indices])

    start = ndim - len(indices)
    end = start + len(indices)
    extra_before, square, extra_after = np.split(shapes, [start, end], axis=1)

    square[np.diag_indices(len(indices))] = 1

    if np.any(extra_before != 1) or np.any(square != 1) or np.any(extra_after != 1):
        return None

    try:
        slices: _SEL_IDS = tuple(
            _as_slice(i.as_array(flattened=True), max_=axis_len)  # if i.ndim > 0 else i
            for i, axis_len in zip(indices, shape)
        )
    except ValueError:
        return None

    return (NewAxis,) * extra_before.shape[1] + slices + (NewAxis,) * extra_after.shape[1]


def _inverse_ix_for_n_lists(indices: _SEL_IDS, shape: _SHAPE) -> _SEL_IDS | None:
    list_indices, lists = zip(*((idx, i) for idx, i in enumerate(indices) if isinstance(i, ListIndex)))

    optimized_lists = _slices_inverse_ix_(lists, shape=shape)

    if optimized_lists is None:
        return None

    it_optimized_lists = iter(optimized_lists)

    return tuple(
        chain.from_iterable(
            takewhile_inclusive(lambda e: e is NewAxis, it_optimized_lists) if idx in list_indices else (i,)
            for idx, i in enumerate(indices)
        )
    )
