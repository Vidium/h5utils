from pathlib import Path
from typing import Generator

import numpy as np
import pytest

import ch5mpy
from ch5mpy import File, H5Array, H5Mode, write_object


def test_array_equal(small_array):
    assert np.array_equal(small_array, [1, 2, 3, 4, 5])


def test_sum(array):
    assert np.sum(array) == 4950


def test_sum_with_initial_value(array):
    assert np.sum(array, initial=1) == 4951


def test_sum_with_output(array):
    out = np.array(0)
    assert np.sum(array, out=out) == 4950


def test_sum_with_initialized_output(array):
    out = np.array(10)
    assert np.sum(array, out=out) == 4950


def test_sum_with_output_and_diff_dtype(array):
    out = np.array(0, dtype=float)
    array += 0.5
    assert np.sum(array, out=out, dtype=int) == 4950


def test_sum_keepdims(array):
    assert np.array_equal(np.sum(array, keepdims=True), [[4950]])


def test_sum_with_axis(array):
    assert np.array_equal(np.sum(array, axis=0), [450, 460, 470, 480, 490, 500, 510, 520, 530, 540])


def test_sum_axis_0(small_large_array):
    with ch5mpy.options(max_memory=str(3 * small_large_array.dtype.itemsize)):
        assert np.array_equal(
            np.sum(small_large_array, axis=0),
            np.array([[60, 63, 66, 69, 72], [75, 78, 81, 84, 87], [90, 93, 96, 99, 102], [105, 108, 111, 114, 117]]),
        )


def test_sum_axis_2(small_large_array):
    with ch5mpy.options(max_memory=str(3 * small_large_array.dtype.itemsize)):
        assert np.array_equal(
            np.sum(small_large_array, axis=2),
            np.array([[10.0, 35.0, 60.0, 85.0], [110.0, 135.0, 160.0, 185.0], [210.0, 235.0, 260.0, 285.0]]),
        )


def test_sum_where(small_large_array):
    assert np.array_equal(
        np.sum(small_large_array, axis=1, where=[True, False, False, True, True]),
        np.array([[30.0, 0.0, 0.0, 42.0, 46.0], [110.0, 0.0, 0.0, 122.0, 126.0], [190.0, 0.0, 0.0, 202.0, 206.0]]),
    )


def test_sum_where_multiple_rows(small_large_array):
    with ch5mpy.options(max_memory=str(40 * small_large_array.dtype.itemsize)):
        assert np.array_equal(
            np.sum(small_large_array, axis=1, where=[True, False, False, True, True]),
            np.array([[30.0, 0.0, 0.0, 42.0, 46.0], [110.0, 0.0, 0.0, 122.0, 126.0], [190.0, 0.0, 0.0, 202.0, 206.0]]),
        )


def test_sum_where_few_elements(small_large_array):
    with ch5mpy.options(max_memory=str(3 * small_large_array.dtype.itemsize)):
        assert np.array_equal(
            np.sum(small_large_array, axis=1, where=[True, False, False, True, True]),
            np.array([[30.0, 0.0, 0.0, 42.0, 46.0], [110.0, 0.0, 0.0, 122.0, 126.0], [190.0, 0.0, 0.0, 202.0, 206.0]]),
        )


def test_all(array):
    assert not np.all(array)


def test_all_axis_keepdims(array):
    assert np.array_equal(
        np.all(array, axis=1, keepdims=True),
        np.array([[False], [True], [True], [True], [True], [True], [True], [True], [True], [True]]),
    )


def test_floor(array):
    array += 0.7
    assert np.array_equal(np.floor(array), np.arange(100.0).reshape((10, 10)))


def test_ceil(array):
    array += 0.2
    assert np.array_equal(np.ceil(array), np.arange(1.0, 101.0).reshape((10, 10)))


# def test_ceil_large(large_array):
#     np.ceil(large_array)


def test_trunc(array):
    array += 0.2
    assert np.array_equal(np.trunc(array), np.arange(100.0).reshape((10, 10)))


def test_prod(array):
    assert np.array_equal(
        np.prod(array, axis=0, where=[True, False, False, False, False, True, True, True, True, True]),
        [0, 1, 1, 1, 1, 6393838623046875, 9585618768101376, 13865696119905399, 19511273389031424, 26853950884211451],
    )


def test_exp(array):
    assert np.array_equal(np.exp(array), np.exp(np.arange(100.0).reshape((10, 10))))


def test_exp_where(small_array):
    out = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    np.exp(small_array, out, where=[True, False, False, True, True])

    assert np.array_equal(out, [np.exp(1), 1.0, 1.0, np.exp(4), np.exp(5)])


def test_expm1(small_array):
    assert np.array_equal(np.expm1(small_array), np.expm1([1, 2, 3, 4, 5]))


def test_isfinite(small_array):
    small_array[0] = np.inf
    assert np.array_equal(np.isfinite(small_array), [False, True, True, True, True])


def test_equal(small_array):
    assert np.all(np.equal(small_array, [1, 2, 3, 4, 5]))
    assert np.array_equal(np.equal(small_array, 2), [False, True, False, False, False])


def test_equal_other_shape(small_array):
    assert np.array_equal(
        small_array == np.array([1, 2, 3, 4, 5]).reshape((-1, 1)),
        np.array(
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ]
        ),
    )


def test_equal_other_shape_str(str_array):
    assert np.array_equal(
        str_array == np.array(["a", "bc", "d", "efg", "h"]).reshape((-1, 1)),
        np.array(
            [
                [True, False, False, False, False],
                [False, True, False, False, False],
                [False, False, True, False, False],
                [False, False, False, True, False],
                [False, False, False, False, True],
            ]
        ),
    )


def test_equal_view(array):
    subarr = array[[[0], [1], [2]], [0, 1]]
    assert np.all(subarr == np.array([[0, 1], [10, 11], [20, 21]]))


def test_equal_view_other_shape(small_array):
    subarr = small_array[[3, 2, 4]]
    assert np.array_equal(
        subarr == np.array([3, 5, 4]).reshape((-1, 1)),
        np.array([[False, True, False], [False, False, True], [True, False, False]]),
    )


def test_equal_2d_array(array):
    assert np.array_equal(array[[[0], [1], [2]], 0], np.array([[0.0], [10.0], [20.0]]))


def test_equal_where(small_array):
    assert np.array_equal(
        np.equal(small_array, [1, 2, 3, 3, 3], where=[True, True, False, False, True]),
        [True, True, False, False, False],
    )


def test_equal_with_broadcast():
    data = np.ones((10, 2))

    with File("h5_equal_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    arr = H5Array(File("h5_equal_array", H5Mode.READ_WRITE)["data"])

    # -----------------------------------------------------
    assert np.all(arr == [1, 1])

    # -----------------------------------------------------
    Path("h5_equal_array").unlink()


def test_greater_element(small_array):
    assert np.array_equal(np.greater(small_array, 3), [False, False, False, True, True])


def test_multiply(small_array):
    assert np.array_equal(np.multiply(small_array, [2, 3, 4, 5, 6]), [2, 6, 12, 20, 30])


def test_multiply_where(small_array):
    assert np.array_equal(
        np.multiply(small_array, [2, 3, 4, 5, 6], where=[True, False, False, True, True]), [2, 2, 3, 20, 30]
    )


def test_isinf(small_array):
    small_array[1] = np.inf
    assert np.array_equal(np.isinf(small_array), [False, True, False, False, False])


def test_isinf_out(small_array):
    small_array[3] = np.inf
    out = np.array([1, 2, 3, 4, 5])
    np.isinf(small_array, where=[True, False, False, True, True], out=out)
    assert np.array_equal(out, [False, 2, 3, True, False])


def test_logical_and(small_array):
    assert np.array_equal(
        np.logical_and(small_array, [True, True, False, False, True]), [True, True, False, False, True]
    )


@pytest.fixture
def repeating_array() -> Generator[H5Array, None, None]:
    data = np.array([[1.0, 2, 1, 1, 2, 1], [2, 1, 0, 1, 1, 2]])

    with File("h5_r_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_r_array", H5Mode.READ_WRITE)["data"])

    Path("h5_r_array").unlink()


def test_unique(repeating_array):
    assert np.array_equal(np.unique(repeating_array), [0, 1, 2])


def test_unique_by_chunks(repeating_array):
    with ch5mpy.options(max_memory=3 * repeating_array.dtype.itemsize):
        assert np.array_equal(np.unique(repeating_array), [0, 1, 2])


def test_unique_with_index(repeating_array):
    with ch5mpy.options(max_memory=3 * repeating_array.dtype.itemsize):
        unique, index = np.unique(repeating_array, return_index=True)
        assert np.array_equal(unique, [0, 1, 2])
        assert np.array_equal(index, [8, 0, 1])


def test_unique_with_counts(repeating_array):
    with ch5mpy.options(max_memory=3 * repeating_array.dtype.itemsize):
        _, counts = np.unique(repeating_array, return_counts=True)
        assert np.array_equal(counts, [1, 7, 4])


def test_unique_not_equal_nan(repeating_array):
    repeating_array[0, 0] = np.nan
    repeating_array[1, 4] = np.nan
    unique = np.unique(repeating_array, equal_nan=False)
    assert np.array_equal(unique, [0, 1, 2, np.nan, np.nan], equal_nan=True)


def test_unique_not_equal_nan_with_counts(repeating_array):
    repeating_array[0, 0] = np.nan
    repeating_array[1, 4] = np.nan
    _, counts = np.unique(repeating_array, return_counts=True, equal_nan=False)
    assert np.array_equal(counts, [1, 5, 4, 1, 1])


def test_in1d_np_in_h5(small_array):
    with ch5mpy.options(max_memory=3 * small_array.dtype.itemsize):
        assert np.array_equal(np.in1d([4, 1, 7], small_array), [True, True, False])


def test_in1d_np_in_h5_invert(small_array):
    with ch5mpy.options(max_memory=3 * small_array.dtype.itemsize):
        assert np.array_equal(np.in1d([4, 1, 7], small_array, invert=True), [False, False, True])


def test_in1d_h5_in_np(small_array):
    with ch5mpy.options(max_memory=3 * small_array.dtype.itemsize):
        assert np.array_equal(np.in1d(small_array, [4, 1, 7]), [True, False, False, True, False])


def test_in1d_h5_in_h5(small_array):
    with ch5mpy.options(max_memory=3 * small_array.dtype.itemsize):
        assert np.array_equal(np.in1d(small_array, small_array), [True, True, True, True, True])


def test_isin_h5_in_np(small_array):
    with ch5mpy.options(max_memory=3 * small_array.dtype.itemsize):
        res = np.isin(small_array, [4, 1, 7])
        assert res.shape == (5,)
        assert np.array_equal(res, [True, False, False, True, False])


def test_isin_np_in_h5(small_array):
    with ch5mpy.options(max_memory=3 * small_array.dtype.itemsize):
        res = np.isin([4, 1, 7], small_array)
        assert res.shape == (3,)
        assert np.array_equal(res, [True, True, False])


def test_isin_h5_in_h5(array, small_array):
    with ch5mpy.options(max_memory=3 * array.dtype.itemsize):
        res = np.isin(array, small_array)
        assert res.shape == (10, 10)

        expected = np.zeros((10, 10), dtype=bool)
        expected[0, [1, 2, 3, 4, 5]] = True
        assert np.array_equal(res, expected)


def test_amax(array):
    assert np.amax(array) == 99
    assert np.array_equal(np.amax(array, axis=1), np.array([9, 19, 29, 39, 49, 59, 69, 79, 89, 99]))


def test_hstack(array):
    assert np.hstack((array, ch5mpy.arange_nd((10, 10)))).shape == (10, 20)


def test_mean(array):
    assert np.mean(array) == 49.5


def test_mean_on_columns(array):
    assert np.array_equal(np.mean(array, axis=1), [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 74.5, 84.5, 94.5])


def test_insert_1D(small_array):
    np.insert(small_array, 2, 0)
    assert np.array_equal(small_array, [1, 2, 0, 3, 4, 5])


def test_insert_1D_start(small_array):
    np.insert(small_array, 0, 0)
    assert np.array_equal(small_array, [0, 1, 2, 3, 4, 5])


def test_insert_1D_end(small_array):
    np.insert(small_array, 5, 0)
    assert np.array_equal(small_array, [1, 2, 3, 4, 5, 0])


def test_insert_1D_negative(small_array):
    np.insert(small_array, -1, 0)
    assert np.array_equal(small_array, [1, 2, 3, 4, 0, 5])


def test_insert_3D(small_large_array):
    np.insert(small_large_array, 1, -1, axis=1)
    assert np.array_equal(
        small_large_array,
        [
            [[0, 1, 2, 3, 4], [-1, -1, -1, -1, -1], [5, 6, 7, 8, 9], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
            [
                [20, 21, 22, 23, 24],
                [-1, -1, -1, -1, -1],
                [25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34],
                [35, 36, 37, 38, 39],
            ],
            [
                [40, 41, 42, 43, 44],
                [-1, -1, -1, -1, -1],
                [45, 46, 47, 48, 49],
                [50, 51, 52, 53, 54],
                [55, 56, 57, 58, 59],
            ],
        ],
    )


def test_delete_1D(small_array):
    np.delete(small_array, 2)
    assert np.array_equal(small_array, [1, 2, 4, 5])


def test_delete_3D(small_large_array):
    np.delete(small_large_array, 1, axis=1)
    assert np.array_equal(
        small_large_array,
        [
            [[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [15, 16, 17, 18, 19]],
            [
                [20, 21, 22, 23, 24],
                [30, 31, 32, 33, 34],
                [35, 36, 37, 38, 39],
            ],
            [
                [40, 41, 42, 43, 44],
                [50, 51, 52, 53, 54],
                [55, 56, 57, 58, 59],
            ],
        ],
    )


def test_ravel(small_large_array) -> None:
    assert np.array_equal(
        np.ravel(small_large_array, order="C"),
        np.array(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
                37,
                38,
                39,
                40,
                41,
                42,
                43,
                44,
                45,
                46,
                47,
                48,
                49,
                50,
                51,
                52,
                53,
                54,
                55,
                56,
                57,
                58,
                59,
            ]
        ),
    )

    assert np.array_equal(
        np.ravel(small_large_array, order="F"),
        np.array(
            [
                0,
                20,
                40,
                5,
                25,
                45,
                10,
                30,
                50,
                15,
                35,
                55,
                1,
                21,
                41,
                6,
                26,
                46,
                11,
                31,
                51,
                16,
                36,
                56,
                2,
                22,
                42,
                7,
                27,
                47,
                12,
                32,
                52,
                17,
                37,
                57,
                3,
                23,
                43,
                8,
                28,
                48,
                13,
                33,
                53,
                18,
                38,
                58,
                4,
                24,
                44,
                9,
                29,
                49,
                14,
                34,
                54,
                19,
                39,
                59,
            ]
        ),
    )


def test_take(small_large_array) -> None:
    assert np.array_equal(np.take(small_large_array, [[0, 1], [2, 3]]), np.array([[0, 1], [2, 3]]))


def test_may_not_share_memory(small_array) -> None:
    assert not np.may_share_memory(small_array, np.array([1, 2, 3]))


def test_may_share_memory(small_array) -> None:
    assert np.may_share_memory(small_array, small_array)


def test_may_share_memory_view(small_array) -> None:
    assert np.may_share_memory(small_array, small_array[1:3])
