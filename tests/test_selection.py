# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from ch5mpy.indexing.list import ListIndex
from ch5mpy.indexing.selection import NewAxis, Selection
from ch5mpy.indexing.slice import FullSlice


# ====================================================
# code
def get_sel(*sel: int | list[Any] | slice | None, shape: tuple[int, ...]) -> Selection:
    return Selection(
        (
            FullSlice.from_slice(s, max_=max_)
            if isinstance(s, slice)
            else NewAxis
            if s is None
            else ListIndex(np.array(s))
            for s, max_ in zip(sel, shape)
        ),
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


def test_selection_should_have_largest_ndims_first():
    assert Selection(
        (ListIndex(np.array(0)), ListIndex(np.array([[0, 1, 2]])), FullSlice(0, 2, 1, 2)), shape=(5, 5, 5)
    ) == Selection(
        (ListIndex(np.array([[0]])), ListIndex(np.array([[0, 1, 2]])), FullSlice(0, 2, 1, 2)), shape=(5, 5, 5)
    )


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
    ],
)
def test_should_cast_shape(previous_selection: Selection, selection: Selection, expected_selection: Selection):
    assert selection.cast_on(previous_selection) == expected_selection


@pytest.mark.parametrize(
    "selection, expected",
    [
        [
            get_sel([[0], [2], [5]], [0, 1], shape=(3, 3)),
            (
                ((0, 0), (0, 0)),
                ((0, 1), (0, 1)),
                ((2, 0), (1, 0)),
                ((2, 1), (1, 1)),
                ((5, 0), (2, 0)),
                ((5, 1), (2, 1)),
            ),
        ],
        [get_sel([[0], [2], [5]], 0, shape=(3, 3)), (((0, 0), (0,)), ((2, 0), (1,)), ((5, 0), (2,)))],
    ],
)
def test_should_iter(selection: Selection, expected):
    assert tuple(selection.iter_indexers()) == expected


@pytest.mark.parametrize(
    "indices, shape, expected",
    [
        (([1, 2, 3, 4, 5],), (6,), (FullSlice(1, 6, 1, 6),)),
        (([[0], [1]], [[2, 3]]), (2, 4), (FullSlice(0, 2, 1, 2), FullSlice(2, 4, 1, 4))),
        (([[3], [2]], [[0, 1]]), (3, 2), (ListIndex(np.array([3, 2])),)),
        ((0, [[[0, 1, 2]]]), (10, 10), (NewAxis, FullSlice(0, 1, 1, 10), FullSlice(0, 3, 1, 10))),
    ],
)
def test_should_optimize(indices, shape, expected: Selection):
    assert get_sel(*indices, shape=shape)._indices == expected


def test_should_get_indexers():
    assert get_sel([[0]], [0], shape=(100, 2)).get_indexers() == (slice(0, 1, 1), slice(0, 1, 1))
