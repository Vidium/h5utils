# coding: utf-8

# ====================================================
# imports
import pytest
import numpy as np
from pathlib import Path

from h5utils import H5Array
from h5utils import File
from h5utils import H5Mode
from h5utils import write_object


# ====================================================
# code
@pytest.fixture
def array() -> H5Array:
    data = np.arange(100).reshape((10, 10))

    with File("h5_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_array", H5Mode.READ_WRITE)["data"])

    Path("h5_array").unlink()


@pytest.fixture
def large_array() -> H5Array:
    data = np.arange(200_000 * 10_000).reshape((200_000, 10_000))

    with File("h5_large_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_large_array", H5Mode.READ_WRITE)["data"])

    Path("h5_large_array").unlink()


def test_should_get_shape(array):
    assert array.shape == (10, 10)


def test_should_get_dtype(array):
    assert array.dtype == np.int64


def test_should_print_repr(array):
    assert (
        repr(array) == "H5Array([[0, 1, 2, ..., 7, 8, 9],\n"
        "         [10, 11, 12, ..., 17, 18, 19],\n"
        "         [20, 21, 22, ..., 27, 28, 29],\n"
        "         ...,\n"
        "         [70, 71, 72, ..., 77, 78, 79],\n"
        "         [80, 81, 82, ..., 87, 88, 89],\n"
        "         [90, 91, 92, ..., 97, 98, 99]], shape=(10, 10), dtype=int64)"
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
    assert str(sub_arr) == "[[20 22 23],\n" " [30 32 33]]\n"


def test_should_get_view(array):
    sub_arr = array[2:4, [0, 2, 3]]
    assert np.array_equal(sub_arr, np.array([[20, 22, 23], [30, 32, 33]]))


def test_should_get_single_value_from_view(array):
    assert array[2:4, [0, 2, 3]][0, 1] == 22
