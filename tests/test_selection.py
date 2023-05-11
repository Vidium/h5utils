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
def get_sel(*sel: int | list[Any] | slice | None) -> Selection:
    return Selection(
        (
            FullSlice.from_slice(s) if isinstance(s, slice) else NewAxis if s is None else ListIndex(np.array(s))
            for s in sel
        )
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


@pytest.mark.parametrize("selection, expected", [[get_sel([1]), ([1],)]])
def test_selection_should_convert_to_lists_and_slices(
    selection: Selection, expected: tuple[list[int] | slice, ...]
) -> None:
    assert _equal(selection.get(), expected)


def test_selection_should_have_largest_ndims_first():
    assert Selection((ListIndex(np.array(0)), ListIndex(np.array([[0, 1, 2]])))) == Selection(
        (ListIndex(np.array([[0]])), ListIndex(np.array([0, 1, 2])))
    )


@pytest.mark.parametrize(
    "selection, expected_shape",
    [
        [get_sel(0), (10,)],
        [get_sel([0]), (1, 10)],
        [get_sel([[0]], slice(0, 10, 1)), (1, 1, 10)],
        [get_sel([[0]]), (1, 1, 10)],
        [get_sel(0, slice(0, 3)), (3,)],
        [get_sel(slice(0, 10), 0), (10,)],
    ],
)
def test_should_compute_shape_2d(selection: Selection, expected_shape):
    assert selection.compute_shape(arr_shape=(10, 10)) == expected_shape


@pytest.mark.parametrize(
    "selection, expected_shape",
    [
        [get_sel(0), (10, 10)],
        [get_sel(0, 0), (10,)],
        [get_sel(0, 0, 0), ()],
        [get_sel([0]), (1, 10, 10)],
        [get_sel(0, [0]), (1, 10)],
        [
            get_sel([0], [0]),
            (
                1,
                10,
            ),
        ],
        [get_sel(0, [[0, 1, 2]]), (1, 3, 10)],
        [get_sel([[0, 1, 2]]), (1, 3, 10, 10)],
        [get_sel([0, 2], [0]), (2, 10)],
        [get_sel([[0]]), (1, 1, 10, 10)],
    ],
)
def test_should_compute_shape_3d(selection: Selection, expected_shape):
    assert selection.compute_shape(arr_shape=(10, 10, 10)) == expected_shape


@pytest.mark.parametrize(
    "previous_selection, selection, expected_selection",
    [
        [get_sel([0, 2], slice(0, 2), [0, 2]), get_sel([0, 1]), get_sel([0, 2], slice(0, 2), [0, 2])],
        [get_sel([0, 2], slice(0, 2), [0, 2]), get_sel(0, 1), get_sel(0, 1, 0)],
        [get_sel([0, 2], slice(0, 2), [0, 2]), get_sel(slice(0, 2), 1), get_sel([0, 2], 1, [0, 2])],
        [get_sel([0, 1, 2], slice(0, 2), [0, 1, 2]), get_sel([0, 2], 1, [0, 1]), get_sel([0, 2], 1, [0, 2], [0, 1])],
        [get_sel(slice(0, 2), slice(1, 3), [0, 1, 2]), get_sel([0, 1], 1, [0, 2]), get_sel([0, 1], 2, [0, 2])],
        [
            get_sel(slice(0, 2), slice(1, 3), [0, 1, 2], [0, 1, 2]),
            get_sel([0, 1], 0, [0, 2], [1, 2]),
            get_sel([0, 1], 1, [0, 2], [0, 2], [1, 2]),
        ],
        [get_sel([0], [0, 1, 2]), get_sel(1), get_sel(0, 1)],
        [get_sel([[0], [1]], [0, 1, 2]), get_sel(1), get_sel([1, 1, 1], [0, 1, 2])],
        [get_sel(0), get_sel(0), get_sel(0, 0)],
        [get_sel([0]), get_sel(0), get_sel(0)],
        [get_sel([[0]]), get_sel(0, 0), get_sel(0)],
        [get_sel([[0], [2], [5]], [[0]]), get_sel(0), get_sel([0], [0])],
        [get_sel(0), get_sel(slice(0, 3)), get_sel(0, slice(0, 3))],
        [get_sel([[0], [1], [2]], 0), get_sel(slice(0, 3), slice(0, 1)), get_sel([[0], [1], [2]], 0)],
        [get_sel(slice(0, 5), None), get_sel(0), get_sel(0, None)],
        [get_sel(slice(0, 5), None), get_sel(0, 0), get_sel(0)],
        [get_sel([[0, 1], [1, 2]], [0, 1]), get_sel(0, 1), get_sel(1, 1)],
        [get_sel([[0], [1], [2], [3], [4]], [[0, 2], [0, 2], [0, 2], [0, 2], [0, 2]]), get_sel(0, 1), get_sel(0, 2)],
    ],
)
def test_should_cast_shape(previous_selection: Selection, selection: Selection, expected_selection: Selection):
    assert selection.cast_on(previous_selection) == expected_selection


@pytest.mark.parametrize(
    "selection, expected",
    [
        [
            get_sel([[0], [2], [5]], [0, 1]),
            (
                ((0, 0), (0, 0)),
                ((0, 1), (0, 1)),
                ((2, 0), (1, 0)),
                ((2, 1), (1, 1)),
                ((5, 0), (2, 0)),
                ((5, 1), (2, 1)),
            ),
        ],
        [get_sel([[0], [2], [5]], 0), (((0, 0), (0,)), ((2, 0), (1,)), ((5, 0), (2,)))],
    ],
)
def test_should_iter(selection: Selection, expected):
    assert tuple(selection.iter_h5((3, 3))) == expected
