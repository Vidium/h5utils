# coding: utf-8
import numpy as np
# ====================================================
# imports
import pytest
from pathlib import Path

from ch5mpy import File
from ch5mpy import H5Mode
from ch5mpy import H5Array
from ch5mpy import write_object


# ====================================================
# code
@pytest.fixture
def str_array() -> H5Array:
    data = ['a', 'b', 'c', 'd', 'e']

    with File("h5_str_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_str_array", H5Mode.READ_WRITE)["data"])

    Path("h5_str_array").unlink()


def test_str_array_dtype(str_array):
    assert str_array.dtype == np.dtype('<U1')


def test_str_array_equals(str_array):
    assert np.array_equal(str_array, ['a', 'b', 'c', 'd', 'e'])


def test_str_array_should_convert_to_numpy_array(str_array):
    np_arr = np.array(str_array)
    assert isinstance(np_arr, np.ndarray)
