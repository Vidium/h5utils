import numpy as np
import numpy.typing as npt

from ch5mpy.indexing.list import ListIndex
from ch5mpy.indexing.slice import FullSlice
from ch5mpy.indexing.special import NewAxis
from ch5mpy.indexing.typing import SELECTION_ELEMENT


def get_valid_indices(
    indices: tuple[SELECTION_ELEMENT, ...], shape: tuple[int, ...], optimize: bool
) -> tuple[SELECTION_ELEMENT, ...]:
    if not len(indices):
        return ()

    if optimize and _gets_whole_dataset(indices):
        return ()

    if not optimize or not all([isinstance(i, ListIndex) for i in indices]):
        index_first_list = next((i for i, e in enumerate(indices) if isinstance(e, ListIndex)), 0)

        return tuple(
            i.squeeze() if isinstance(i, ListIndex) and idx > index_first_list else i for idx, i in enumerate(indices)
        )

    optimized_indices = _slices_inverse_ix_(*(np.array(i) for i in indices), shape=shape)

    if optimized_indices is None:
        index_first_list = next((i for i, e in enumerate(indices) if isinstance(e, ListIndex)), 0)

        return tuple(
            i.squeeze() if isinstance(i, ListIndex) and idx > index_first_list else i for idx, i in enumerate(indices)
        )

    return optimized_indices


def _gets_whole_dataset(index: tuple[SELECTION_ELEMENT, ...]) -> bool:
    return all(isinstance(i, FullSlice) and i.is_whole_axis for i in index)


def _as_slice(array: npt.NDArray[np.int_], max_: int) -> FullSlice:
    if len(array) == 0:
        return FullSlice(0, 0, 1, max_=max_)

    if len(array) == 1:
        return FullSlice(array[0], array[0] + 1, 1, max_=max_)

    step = array[1] - array[0]
    miss_steps = np.where(np.ediff1d(array) != step)[0]

    if len(miss_steps):
        raise ValueError

    return FullSlice(array[0], array[-1] + step, step, max_=max_)


def _slices_inverse_ix_(*arrays: npt.NDArray[np.int_], shape: tuple[int, ...]) -> tuple[SELECTION_ELEMENT, ...] | None:
    """
    From indexing arrays, return (if possible) the tuple of slices that would lead to the same indexing since indexing
    from slices leads to much fewer read operations.
    Also, return the number of NewAxes to add before and after the tuple of slices so as to maintain the resulting
    array's dimensions.

    This is effectively the inverse of the numpy.ix_ function, with indexing arrays converted to slices.
    """
    ndim = arrays[0].ndim
    if len(arrays) > ndim:
        return None

    try:
        shapes = np.array([a.shape for a in arrays])
    except ValueError:
        return None

    start = ndim - len(arrays)
    end = start + len(arrays)
    extra_before, square, extra_after = shapes[:, 0:start], shapes[:, start:end], shapes[:, end:]

    if np.any(extra_before != 1) or np.any(extra_after != 1):
        return None

    square[np.diag_indices(len(arrays))] = 1

    if np.any(square != 1):
        return None

    try:
        slices = tuple(_as_slice(arr.flatten(), max_=axis_len) for arr, axis_len in zip(arrays, shape))
    except ValueError:
        return None

    return (NewAxis,) * extra_before.shape[1] + slices + (NewAxis,) * extra_after.shape[1]
