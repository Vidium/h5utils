# coding: utf-8

# ====================================================
# imports
import numpy as np
import pytest

import ch5mpy


# ====================================================
# code
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
    assert isinstance(np.asarray(array), np.ndarray)
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
    assert np.array_equal(
        s, np.array([450, 460, 470, 480, 490, 500, 510, 520, 530, 540])
    )


def test_should_add_inplace(array):
    array += 1
    assert np.array_equal(array, np.arange(100).reshape((10, 10)) + 1)


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
    assert np.array_equal(subset, np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                            [20, 21, 22, 23, 24, 25, 26, 27, 28, 29]]))


def test_should_subset_from_2_boolean_arrays(array):
    subset = array[[True, False, True, False, False, False, False, False, False, False],
                   [True, False, True, False, False, False, False, False, False, False]]
    assert np.array_equal(subset, np.array([0, 22]))


def test_should_set_value_in_array(array):
    array[5, 7] = -1
    assert array[5, 7] == -1


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
    small_large_array.MAX_MEM_USAGE = str(3 * small_large_array.dtype.itemsize)
    assert 10 in small_large_array


def test_should_not_find_missing_value_in_array(small_large_array):
    small_large_array.MAX_MEM_USAGE = str(3 * small_large_array.dtype.itemsize)
    assert 1000 not in small_large_array


def test_subset_1d(array):
    subset = array[0]
    assert subset.ndim == 1


def test_subset_1d_column(array):
    subset = array[:, 0]
    assert subset.shape == (10,)


def test_subset_2d(array):
    subset = array[[0, 2], [[0]]]
    assert subset.ndim == 2


def test_iter_chunks_str_array(str_array):
    _, chunk = list(str_array.iter_chunks())[0]
    assert np.issubdtype(chunk.dtype, str)


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
    assert repr(subarr) == "H5Array([[0.0],\n" \
                           "         [20.0],\n" \
                           "         [50.0]], shape=(3, 1), dtype=float64)"


def test_array_should_get_array_in_1d_from_slice(array):
    subarr = array[:3]
    subsubarr = subarr[0]
    assert isinstance(subsubarr, ch5mpy.H5Array)
    assert subsubarr.shape == (10,)


def test_array_type_casting(array):
    assert array.astype(int).dtype == np.int64


def test_array_str_type_casting(str_array, array):
    assert str_array.astype(str).dtype == np.dtype('<U3')
    assert array.astype(str).dtype == np.dtype('<U32')


@pytest.mark.xfail
def test_array_str_cast_int_should_fail(str_array):
    _ = str_array.astype(int)[:]


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

    ref = np.arange(100.).reshape((10, 10))
    ref[0, [0, 1]] = [-1, -2]

    assert np.array_equal(array, ref)


def test_setitem_str(str_array):
    str_array[[0, 1]] = np.array(['A', 'BBBB'])
    assert np.array_equal(str_array, ['A', 'BBBB', 'd', 'efg', 'h'])
    assert str_array.dtype == np.dtype('<U4')
