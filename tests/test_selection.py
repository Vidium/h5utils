# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import pytest
import numpy as np

from typing import Any

from ch5mpy.h5array.indexing.list import ListIndex
from ch5mpy.h5array.indexing.slice import FullSlice
from ch5mpy.h5array.indexing.selection import Selection


# ====================================================
# code
def get_sel(*sel: int | list[Any] | slice) -> Selection:
    return Selection((FullSlice.from_slice(s) if isinstance(s, slice) else ListIndex(np.array(s)) for s in sel))


def test_selection_should_have_largest_ndims_first():
    assert Selection((ListIndex(np.array(0)), ListIndex(np.array([[0, 1, 2]])))) == \
           Selection((ListIndex(np.array([[0]])), ListIndex(np.array([0, 1, 2]))))


@pytest.mark.parametrize(
    'selection, expected_shape',
    [
        [get_sel([0]), (1, 10)],
        [get_sel([[0]], slice(0, 10, 1)), (1, 1, 10)],
        [get_sel([[0]]), (1, 1, 10)],
        [get_sel(0, slice(0, 3)), (3,)]
    ]
)
def test_should_compute_shape_2d(selection: Selection, expected_shape):
    assert selection.compute_shape(arr_shape=(10, 10)) == expected_shape


@pytest.mark.parametrize(
    'selection, expected_shape',
    [
        [get_sel(0), (10, 10)],
        [get_sel(0, 0), (10,)],
        [get_sel(0, 0, 0), ()],
        [get_sel([0]), (1, 10, 10)],
        [get_sel(0, [0]), (1, 10)],
        [get_sel([0], [0]), (1, 10,)],
        [get_sel(0, [[0, 1, 2]]), (1, 3, 10)],
        [get_sel([[0, 1, 2]]), (1, 3, 10, 10)],
        [get_sel([0, 2], [0]), (2, 10)],
        [get_sel([[0]]), (1, 1, 10, 10)]
    ]
)
def test_should_compute_shape_3d(selection: Selection, expected_shape):
    assert selection.compute_shape(arr_shape=(10, 10, 10)) == expected_shape


@pytest.mark.parametrize(
    'previous_selection, selection, expected_selection',
    [
        [get_sel([0, 2], slice(0, 2), [0, 2]), get_sel([0, 1]),
         get_sel([0, 2], slice(0, 2), [0, 2])],
        [get_sel([0, 2], slice(0, 2), [0, 2]), get_sel(0, 1),
         get_sel(0, 1, 0)],
        [get_sel([0, 2], slice(0, 2), [0, 2]), get_sel(slice(0, 2), 1),
         get_sel([0, 2], 1, [0, 2])],
        [get_sel([0, 1, 2], slice(0, 2), [0, 1, 2]), get_sel([0, 2], 1, [0, 1]),
         get_sel([0, 2], 1, [0, 2], [0, 1])],
        [get_sel(slice(0, 2), slice(1, 3), [0, 1, 2]), get_sel([0, 1], 1, [0, 2]),
         get_sel([0, 1], 2, [0, 2])],
        [get_sel(slice(0, 2), slice(1, 3), [0, 1, 2], [0, 1, 2]), get_sel([0, 1], 0, [0, 2], [1, 2]),
         get_sel([0, 1], 1, [0, 2], [0, 2], [1, 2])],
        [get_sel([0], [0, 1, 2]), get_sel(1),
         get_sel(0, 1)],
        [get_sel([[0], [1]], [0, 1, 2]), get_sel(1),
         get_sel([1], [0, 1, 2])],
        [get_sel(0,), get_sel(0),
         get_sel(0, 0)],
        [get_sel([0]), get_sel(0),
         get_sel(0)],
        [get_sel([[0]]), get_sel(0, 0),
         get_sel(0)],
        [get_sel([[0], [2], [5]], [[0]]), get_sel(0),
         get_sel([0], [0])],
        [get_sel(0), get_sel(slice(0, 3)),
         get_sel(0, slice(0, 3))],
        [get_sel([[0], [1], [2]], 0), get_sel(slice(0, 3), slice(0, 1)),
         get_sel([[0], [1], [2]], 0)]
    ]
)
def test_should_cast_shape(previous_selection: Selection, selection: Selection, expected_selection: Selection):
    assert selection.cast_on(previous_selection) == expected_selection


@pytest.mark.parametrize(
    'selection, expected',
    [
        [get_sel([[0], [1], [2]], [0, 1]), (((0, 0), (0, 0)), ((0, 1), (1, 1)), ((1, 0), (2, 2)),
                                            ((1, 1), (3, 3)), ((2, 0), (4, 4)), ((2, 1), (5, 5)))],
        [get_sel([[0], [2], [5]], 0), (((0, 0), (0, slice(None))),
                                       ((2, 0), (1, slice(None))),
                                       ((5, 0), (2, slice(None))))]
    ]
)
def test_should_iter(selection: Selection, expected):
    assert tuple(selection.iter_h5(2)) == expected
