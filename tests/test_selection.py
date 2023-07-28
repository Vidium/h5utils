from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ch5mpy.indexing import FullSlice, ListIndex, as_indexer
from ch5mpy.indexing.selection import Selection


def get_sel(*sel: int | list[Any] | slice | None, shape: tuple[int, ...]) -> Selection:
    return Selection(
        (as_indexer(s, max=max) for s, max in zip(sel, shape)),
        shape=shape,
    )


def _equal(s1: tuple[Any, ...], s2: tuple[Any, ...]) -> bool:
    if not len(s1) == len(s2):
        return False

    for e1, e2 in zip(s1, s2):
        if isinstance(e1, np.ndarray) or isinstance(e2, np.ndarray):
            if not np.array_equal(e1, e2):
                return False

        elif e1 != e2:
            return False

    return True


@pytest.mark.parametrize(
    "selection, expected_shape",
    [
        [get_sel(0, shape=(10, 10)), (10,)],
        [get_sel([0], shape=(10, 10)), (1, 10)],
        [get_sel([[0]], slice(0, 10, 1), shape=(10, 10)), (1, 1, 10)],
        [get_sel([[0]], shape=(10, 10)), (1, 1, 10)],
        [get_sel(0, slice(0, 3), shape=(10, 10)), (3,)],
        [get_sel(slice(0, 10), 0, shape=(10, 10)), (10,)],
    ],
)
def test_should_compute_shape_2d(selection: Selection, expected_shape):
    get_sel([[0]], slice(0, 10, 1), shape=(10, 10))
    assert selection.out_shape == expected_shape


@pytest.mark.parametrize(
    "selection, expected_shape",
    [
        [get_sel(0, shape=(10, 10, 10)), (10, 10)],
        [get_sel(0, 0, shape=(10, 10, 10)), (10,)],
        [get_sel(0, 0, 0, shape=(10, 10, 10)), ()],
        [get_sel([0], shape=(10, 10, 10)), (1, 10, 10)],
        [get_sel(0, [0], shape=(10, 10, 10)), (1, 10)],
        [get_sel([0], [0], shape=(10, 10, 10)), (1, 10)],
        [get_sel(0, [[0, 1, 2]], shape=(10, 10, 10)), (1, 3, 10)],
        [get_sel([[0, 1, 2]], shape=(10, 10, 10)), (1, 3, 10, 10)],
        [get_sel([0, 2], [0], shape=(10, 10, 10)), (2, 10)],
        [get_sel([[0]], shape=(10, 10, 10)), (1, 1, 10, 10)],
    ],
)
def test_should_compute_shape_3d(selection: Selection, expected_shape):
    get_sel([[0, 1, 2]], shape=(10, 10, 10))
    assert selection.out_shape == expected_shape


@pytest.mark.parametrize(
    "previous_selection, selection, expected_selection",
    [
        [
            get_sel([0, 2], slice(0, 2), [0, 2], shape=(100, 100, 100)),
            get_sel([0, 1], shape=(2, 2)),
            get_sel([0, 2], slice(0, 2), [0, 2], shape=(100, 100, 100)),
        ],
        [
            get_sel([0, 2], slice(0, 2), [0, 2], shape=(100, 100, 100)),
            get_sel(0, 1, shape=(2, 2)),
            get_sel(0, 1, 0, shape=(100, 100, 100)),
        ],
        [
            get_sel([0, 2], slice(0, 2), [0, 2], shape=(100, 100, 100)),
            get_sel(slice(0, 2), 1, shape=(2, 2)),
            get_sel([0, 2], 1, [0, 2], shape=(100, 100, 100)),
        ],
        [
            get_sel([0, 1, 2], slice(0, 2), [0, 1, 2], shape=(100, 100, 100, 100)),
            get_sel([0, 2], 1, [0, 1], shape=(3, 2, 100)),
            get_sel([0, 2], 1, [0, 2], [0, 1], shape=(100, 100, 100, 100)),
        ],
        [
            get_sel(slice(0, 2), slice(1, 3), [0, 1, 2], shape=(100, 100, 100)),
            get_sel([0, 1], 1, [0, 2], shape=(2, 2, 3)),
            get_sel([0, 1], 2, [0, 2], shape=(100, 100, 100)),
        ],
        [
            get_sel(slice(0, 2), slice(1, 3), [0, 1, 2], [0, 1, 2], shape=(100, 100, 100, 100, 100)),
            get_sel([0, 1], 0, [0, 2], [1, 2], shape=(2, 2, 3, 100)),
            get_sel([0, 1], 1, [0, 2], [0, 2], [1, 2], shape=(100, 100, 100, 100, 100)),
        ],
        [
            get_sel([0], [0, 1, 2], shape=(100, 100, 100)),
            get_sel(1, shape=(3, 100)),
            get_sel(0, 1, shape=(100, 100, 100)),
        ],
        [
            get_sel([[0], [1]], [0, 1, 2], shape=(100, 100, 100)),
            get_sel(1, shape=(2, 3, 100)),
            get_sel(1, slice(0, 3), shape=(100, 100, 100)),
        ],
        [get_sel(0, shape=(100, 100, 100)), get_sel(0, shape=(100, 100)), get_sel(0, 0, shape=(100, 100, 100))],
        [get_sel([0], shape=(100, 100, 100)), get_sel(0, shape=(1, 100, 100)), get_sel(0, shape=(100, 100, 100))],
        [
            get_sel([[0]], shape=(100, 100, 100)),
            get_sel(0, 0, shape=(1, 1, 100, 100)),
            get_sel(0, shape=(100, 100, 100)),
        ],
        [
            get_sel([[0], [2], [5]], [[0]], shape=(100, 100, 100)),
            get_sel(0, shape=(3, 1, 100)),
            get_sel([0], [0], shape=(100, 100, 100)),
        ],
        [
            get_sel(0, shape=(100, 100, 100)),
            get_sel(slice(0, 3), shape=(100, 100)),
            get_sel(0, slice(0, 3), shape=(100, 100, 100)),
        ],
        [
            get_sel([[0], [1], [2]], 0, shape=(100, 100, 100)),
            get_sel(slice(0, 3), slice(0, 1), shape=(3, 1, 100)),
            get_sel([[0], [1], [2]], 0, shape=(100, 100, 100)),
        ],
        [
            get_sel(slice(0, 5), None, shape=(100, 100, 100)),
            get_sel(0, shape=(5, 1, 100, 100)),
            get_sel(0, None, shape=(100, 100, 100)),
        ],
        [
            get_sel(slice(0, 5), None, shape=(100, 100, 100)),
            get_sel(0, 0, shape=(5, 1, 100, 100)),
            get_sel(0, shape=(100, 100, 100)),
        ],
        [
            get_sel([[0, 1], [1, 2]], [0, 1], shape=(100, 100, 100)),
            get_sel(0, 1, shape=(2, 2, 100)),
            get_sel(1, 1, shape=(100, 100, 100)),
        ],
        [
            get_sel([[0], [1], [2], [3], [4]], [[0, 2], [0, 2], [0, 2], [0, 2], [0, 2]], shape=(100, 100, 100)),
            get_sel(0, 1, shape=(5, 2, 100)),
            get_sel(0, 2, shape=(100, 100, 100)),
        ],
        [
            get_sel([0, 2, 1, 3], shape=(100, 100, 100)),
            get_sel(slice(None), [0, 2, 1], shape=(4, 100, 100)),
            get_sel([[0], [2], [1], [3]], [0, 2, 1], shape=(100, 100, 100)),
        ],
        [get_sel([], shape=(5, 5)), get_sel(slice(0, 3), shape=(5,)), get_sel([], shape=(5, 5))],
        [
            get_sel([[4], [0], [2]], [0], shape=(10, 2)),
            get_sel(slice(0, 2), shape=(3, 1)),
            get_sel([[4], [0]], [[0], [0]], shape=(10, 2)),
        ],
    ],
)
def test_should_cast_shape(previous_selection: Selection, selection: Selection, expected_selection: Selection):
    assert selection.cast_on(previous_selection) == expected_selection


@pytest.mark.parametrize(
    "selection, can_reorder, expected",
    [
        [
            get_sel([[0], [2], [5]], [0, 1], shape=(10, 10)),
            True,
            (
                ((0, 0), (0, 0)),
                ((0, 1), (0, 1)),
                ((2, 0), (1, 0)),
                ((2, 1), (1, 1)),
                ((5, 0), (2, 0)),
                ((5, 1), (2, 1)),
            ),
        ],
        [get_sel([[0], [2], [5]], 0, shape=(10, 10)), True, (((np.array([0, 2, 5]), 0), (slice(None), 0)),)],
        [get_sel([54, 50, 52], shape=(100, 10)), False, (((54,), (0,)), ((50,), (1,)), ((52,), (2,)))],
    ],
)
def test_should_iter(selection: Selection, can_reorder, expected):
    indexers = tuple(selection.iter_indexers(can_reorder=can_reorder))

    assert len(indexers) == len(expected)
    for (dest_idx, load_idx), (exp_dest_idx, exp_load_idx) in zip(indexers, expected):
        assert _equal(dest_idx, exp_dest_idx) and _equal(load_idx, exp_load_idx)


@pytest.mark.parametrize(
    "indices, shape, expected",
    [
        (([1, 2, 3, 4, 5],), (6,), (FullSlice(1, 6, 1, 6),)),
        (([[0], [1]], [[2, 3]]), (2, 4), (FullSlice(0, 2, 1, 2), FullSlice(2, 4, 1, 4))),
        (([[2], [1]], [[0, 1]]), (3, 2), (ListIndex(np.array([2, 1]), 3),)),
        # ((0, [[[0, 1, 2]]]), (10, 10), (NewAxis, SingleIndex(0, 10), FullSlice(0, 3, 1, 10))),  # FIXME
        ((slice(None), [1, 0]), (10, 2), (FullSlice(0, 10, 1, 10), ListIndex(np.array([1, 0]), 2))),
        (([[0], [1], [2], [3]], [0, 3, 1]), (4, 4), (FullSlice(0, 4, 1, 4), ListIndex(np.array([0, 3, 1]), 4))),
        (([[1]], [[0, 1]]), (4, 4), (FullSlice(1, 2, 1, 4), FullSlice(0, 2, 1, 4))),
    ],
)
def test_should_optimize(indices, shape, expected: Selection):
    assert get_sel(*indices, shape=shape)._indices == expected


@pytest.mark.parametrize(
    "sel, shape",
    [(get_sel(slice(5), [], shape=(10, 10)), (5, 0)), (get_sel([[0], [1], [2]], [], shape=(10, 10)), (3, 0))],
)
def test_should_get_shape(sel, shape):
    assert sel.out_shape == shape
