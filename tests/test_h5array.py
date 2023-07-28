from pathlib import Path

import numpy as np
import pytest

import ch5mpy


def test_should_get_shape(array):
    assert array.shape == (10, 10)


def test_should_get_dtype(array):
    assert array.dtype == np.float64


def test_should_print_repr(array):
    assert (
        repr(array) == "H5Array([[0.0, 1.0, 2.0, ..., 7.0, 8.0, 9.0],\n"
        "         [10.0, 11.0, 12.0, ..., 17.0, 18.0, 19.0],\n"
        "         [20.0, 21.0, 22.0, ..., 27.0, 28.0, 29.0],\n"
        "         ...,\n"
        "         [70.0, 71.0, 72.0, ..., 77.0, 78.0, 79.0],\n"
        "         [80.0, 81.0, 82.0, ..., 87.0, 88.0, 89.0],\n"
        "         [90.0, 91.0, 92.0, ..., 97.0, 98.0, 99.0]], shape=(10, 10), dtype=float64)"
    )


def test_should_convert_to_numpy_array(array):
    assert type(np.asarray(array)) == np.ndarray
    assert np.array_equal(np.asarray(array), np.arange(100).reshape((10, 10)))


def test_should_pass_numpy_ufunc(array):
    arr_2 = np.multiply(array, 2)
    assert np.array_equal(arr_2, np.arange(100).reshape((10, 10)) * 2)


def test_should_work_with_magic_operations(array):
    arr_2 = array + 2
    assert np.array_equal(arr_2, np.arange(100).reshape((10, 10)) + 2)


def test_should_sum_all_values(array):
    s = np.sum(array)
    assert s == 4950


def test_should_sum_along_axis(array):
    s = np.sum(array, axis=0)
    assert np.array_equal(s, np.array([450, 460, 470, 480, 490, 500, 510, 520, 530, 540]))


def test_should_add_inplace(array):
    array += 1
    assert np.array_equal(array, np.arange(100).reshape((10, 10)) + 1)


def test_should_add_to_view(array):
    view = array[:, 0]
    res = view + 1

    assert type(res) == np.ndarray
    assert np.array_equal(res, np.array([1, 11, 21, 31, 41, 51, 61, 71, 81, 91]))


def test_should_add_inplace_to_view(array):
    v = array[1:3, 5:8]
    v += 1

    assert np.array_equal(
        array,
        np.array(
            [
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 16, 17, 18, 18, 19],
                [20, 21, 22, 23, 24, 26, 27, 28, 28, 29],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            ]
        ),
    )


def test_should_add_inplace_to_view_2(array):
    array[[0, 2, 3], 5:7] += 1

    assert np.array_equal(
        array,
        np.array(
            [
                [0, 1, 2, 3, 4, 6, 7, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, 22, 23, 24, 26, 27, 27, 28, 29],
                [30, 31, 32, 33, 34, 36, 37, 37, 38, 39],
                [40, 41, 42, 43, 44, 45, 46, 47, 48, 49],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            ]
        ),
    )


# def test_large_array(large_array):
#     large_array += 2


def test_should_get_single_element(array):
    assert array[1, 2] == 12


def test_should_get_whole_dset(array):
    assert np.array_equal(array[:], array)
    assert np.array_equal(array[()], array)


def test_should_print_view_repr(array):
    sub_arr = array[2:4, [0, 2, 3]]
    assert str(sub_arr) == "[[20.0 22.0 23.0],\n" " [30.0 32.0 33.0]]\n"


def test_should_get_view(array):
    sub_arr = array[2:4, [0, 2, 3]]
    assert np.array_equal(sub_arr, np.array([[20, 22, 23], [30, 32, 33]]))


def test_should_get_view_from_view(array):
    sub_arr = array[2:4, [0, 2, 3]]
    sub_sub_arr = sub_arr[1, [1, 2]]
    assert np.array_equal(sub_sub_arr, np.array([32, 33]))


def test_should_get_single_value_from_view(array):
    assert array[2:4, [0, 2, 3]][0, 1] == 22


def test_should_subset_from_boolean_array(array):
    subset = array[np.array([True, False, True, False, False, False, False, False, False, False])]
    assert np.array_equal(subset, np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]))


def test_should_subset_from_2_boolean_arrays(array):
    subset = array[
        [True, False, True, False, False, False, False, False, False, False],
        [True, False, True, False, False, False, False, False, False, False],
    ]
    assert np.array_equal(subset, np.array([0, 22]))


def test_should_set_value_in_array(array):
    array[5, 7] = -1
    assert array[5, 7] == -1


def test_should_set_values_in_array_in_correct_order(array):
    array[:, [2, 3]][[4, 0, 2]] = np.array([[-1, -2], [-3, -4], [-5, -6]])
    assert np.array_equal(
        array,
        np.array(
            [
                [0, 1, -3, -4, 4, 5, 6, 7, 8, 9],
                [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                [20, 21, -5, -6, 24, 25, 26, 27, 28, 29],
                [30, 31, 32, 33, 34, 35, 36, 37, 38, 39],
                [40, 41, -1, -2, 44, 45, 46, 47, 48, 49],
                [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
                [60, 61, 62, 63, 64, 65, 66, 67, 68, 69],
                [70, 71, 72, 73, 74, 75, 76, 77, 78, 79],
                [80, 81, 82, 83, 84, 85, 86, 87, 88, 89],
                [90, 91, 92, 93, 94, 95, 96, 97, 98, 99],
            ]
        ),
    )


def test_should_set_value_in_view(array):
    sub_arr = array[2:4, [0, 2, 3]]
    sub_arr[1, [1, 2]] = [-2, -3]
    assert np.array_equal(array[3, [2, 3]], [-2, -3])


def test_apply_all_function(array):
    assert not np.all(array)

    array += 1
    assert np.all(array)


def test_apply_any_function(array):
    assert np.any(array)


def test_should_find_value_in_array(small_large_array):
    with ch5mpy.options(max_memory=str(3 * small_large_array.dtype.itemsize)):
        assert 10 in small_large_array


def test_should_not_find_missing_value_in_array(small_large_array):
    with ch5mpy.options(max_memory=str(3 * small_large_array.dtype.itemsize)):
        assert 1000 not in small_large_array


def test_should_find_value_in_view(small_array) -> None:
    view = small_array[2:]
    assert 4 in view


def test_should_not_find_missing_value_in_view(small_array) -> None:
    view = small_array[2:]
    assert 1 not in view


def test_subset_0d(array):
    subset = array[[]]
    assert subset.shape == (0, 10)
    assert repr(subset) == "H5Array([], shape=(0, 10), dtype=float64)"


def test_subset_1d_0d(array):
    subset = array[np.ix_([0, 1, 2], [])]
    assert subset.shape == (3, 0)
    assert subset.size == 0
    assert repr(subset) == "H5Array([], shape=(3, 0), dtype=float64)"


def test_subset_1d(array):
    subset = array[0]
    assert subset.ndim == 1


def test_subset_1d_column(array):
    subset = array[:, 0]
    assert subset.shape == (10,)


def test_subset_2d(array):
    subset = array[[0, 2], [[0]]]
    assert subset.ndim == 2


def test_view_should_convert_to_numpy(array):
    subset = array[[0, 2], [[0]]]
    subset = subset.copy()
    assert np.array_equal(subset, [[0.0, 20.0]])


def test_view_should_convert_to_numpy_2(small_large_array):
    subset = small_large_array[[[0, 1], [1, 0]], [[3, 2], [1, 0]], [[0, 1], [2, 3]]]
    subset = subset.copy()
    assert np.array_equal(subset, [[15, 31], [27, 3]])


def test_view_as_object_dtype_should_convert_to_numpy(str_array):
    subset = str_array[0:1].astype(object)
    subset = subset.copy()
    assert np.array_equal(subset, [b"a"])


@pytest.mark.parametrize("subset, shape", [[(slice(None), 0), (0,)], [(None, slice(None), 0, None), (1, 0, 1)]])
def test_subset_from_empty_array(empty_array, subset, shape):
    subset = empty_array[subset]
    assert subset.shape == shape


def test_empty_array_should_convert_to_numpy_array(empty_array):
    assert np.array_equal(np.array(empty_array), np.empty((0, 1)))


@pytest.mark.parametrize("subset, array", [[(slice(None), 0), np.empty(0)], [[], np.empty((0, 1))]])
def test_empty_array_subset_should_convert_to_numpy_array(empty_array, subset, array):
    subset = empty_array[subset]
    assert np.array_equal(np.array(subset), array)


def test_array_subset_ix(array):
    assert array[np.ix_([5], [5])] == 55


def test_array_subset_2d(array):
    subarr = array[[0]]
    assert subarr.ndim == 2
    assert repr(subarr) == "H5Array([[0.0, 1.0, 2.0, ..., 7.0, 8.0, 9.0]], shape=(1, 10), dtype=float64)"
    assert subarr[0].ndim == 1


def test_array_subset_3d(array):
    subarr = array[[[0]]]
    assert subarr.ndim == 3
    assert subarr[0].ndim == 2
    assert subarr[0, 0].ndim == 1
    assert repr(subarr) == "H5Array([[[0.0, 1.0, 2.0, ..., 7.0, 8.0, 9.0]]], shape=(1, 1, 10), dtype=float64)"


def test_array_should_get_one_element(array):
    subarr = array[0, 0]
    assert isinstance(subarr, np.float_)


def test_array_should_get_array_of_one_element_in_1d(array):
    subarr = array[[0], [0]]
    assert isinstance(subarr, ch5mpy.H5Array)
    assert subarr.shape == (1,)


def test_array_should_get_array_of_one_element_in_2d(array):
    subarr = array[[0], [[0]]]
    assert isinstance(subarr, ch5mpy.H5Array)
    assert subarr.shape == (1, 1)


def test_array_should_get_array_of_multiple_elements_in_2d(array):
    subarr = array[[[0], [2], [5]], [[0]]]
    assert isinstance(subarr, ch5mpy.H5Array)
    assert subarr.shape == (3, 1)
    assert subarr[0].shape == (1,)
    assert repr(subarr) == "H5Array([[0.0],\n" "         [20.0],\n" "         [50.0]], shape=(3, 1), dtype=float64)"


def test_array_should_get_array_in_1d_from_slice(array):
    subarr = array[:3]
    subsubarr = subarr[0]
    assert isinstance(subsubarr, ch5mpy.H5Array)
    assert subsubarr.shape == (10,)


def test_array_type_casting(array):
    assert array.astype(int).dtype == np.int64


def test_view_type_casting(array):
    subarr = array[:5, :5]
    assert subarr.astype(int).shape == (5, 5)


def test_subset_newaxis(small_array):
    new_array = small_array[:, None]
    assert new_array.shape == (5, 1)


def test_subset_newaxis_shape(small_array):
    new_array = small_array[:, None]

    subarr = new_array[0]
    assert subarr.shape == (1,)

    subsubarr = subarr[0]
    assert isinstance(subsubarr, np.float_)


def test_subset_multiple_newaxis_repr(small_array):
    new_array = small_array[None, None]

    assert new_array.shape == (1, 1, 5)
    assert repr(new_array) == "H5Array([[[1.0, 2.0, 3.0, 4.0, 5.0]]], shape=(1, 1, 5), dtype=float64)"


def test_newaxis_to_numpy_array(small_array):
    new_array = small_array[:, None]
    assert np.array(new_array).shape == (5, 1)


def test_setitem(array):
    array[0, [0, 1]] = np.array([[-1, -2]])

    ref = np.arange(100.0).reshape((10, 10))
    ref[0, [0, 1]] = [-1, -2]

    assert np.array_equal(array, ref)


def test_setitem_row(array):
    array[[[1]], [[0, 1]]] = np.array([[-1, -2]])


def test_setitem_column(array):
    v = -1 * np.ones(5)[:, None]
    array[[[0], [1], [2], [3], [4]], [[0]]] = v


def test_get_in_random_order(array):
    v = array[np.ix_([5, 1, 2], [9, 8, 7])]
    a = np.array(v)
    assert np.array_equal(a, [[59, 58, 57], [19, 18, 17], [29, 28, 27]])


def test_get_in_random_order_1d(small_array):
    v = small_array[[1, 3, 4, 0, 2]]
    a = np.array(v)
    assert np.array_equal(a, [2, 4, 5, 1, 3])


def test_get_in_random_order_2d(array):
    v = array[np.ix_([1, 4, 1, 3, 1, 2], [2, 1, 0])]
    a = np.array(v)
    assert np.array_equal(
        a,
        np.array(
            [
                [12.0, 11.0, 10.0],
                [42.0, 41.0, 40.0],
                [12.0, 11.0, 10.0],
                [32.0, 31.0, 30.0],
                [12.0, 11.0, 10.0],
                [22.0, 21.0, 20.0],
            ]
        ),
    )


def test_inplace_type_casting(array: ch5mpy.H5Array):
    array.astype(str, inplace=True)
    assert array.dtype == np.dtype("<U4")


# FIXME
@pytest.mark.xfail
def test_repr_3d_array():
    path = Path("f.h5")
    if path.exists():
        path.unlink()

    array = ch5mpy.ones((10, 10, 10), "array", "f.h5")
    assert (
        repr(array) == "H5Array([[[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          ...,\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]],\n"
        "\n"
        "         [[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          ...,\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]],\n"
        "\n"
        "         [[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          ...,\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]],\n"
        "\n"
        "         ...,\n"
        "\n"
        "         [[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          ...,\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]],\n"
        "\n"
        "         [[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          ...,\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]],\n"
        "\n"
        "         [[1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          ...,\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0],\n"
        "          [1.0, 1.0, 1.0, ..., 1.0, 1.0, 1.0]]], shape=(10, 10, 10), dtype=float64)"
    )
    path.unlink()


def test_can_flatten(small_large_array: ch5mpy.H5Array) -> None:
    assert small_large_array.flatten().shape == (60,)


def test_can_hash(small_array: ch5mpy.H5Array) -> None:
    hash1 = hash(small_array)
    small_array[0] = -1
    hash2 = hash(small_array)
    assert hash1 != hash2
