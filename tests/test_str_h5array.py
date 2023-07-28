import numpy as np
import pytest


def test_str_array_dtype(str_array):
    assert str_array.dtype == np.dtype("<U3")


def test_str_array_equals(str_array):
    assert np.array_equal(str_array, ["a", "bc", "d", "efg", "h"])


def test_str_array_should_convert_to_numpy_array(str_array):
    np_arr = np.array(str_array)
    assert type(np_arr) == np.ndarray


def test_str_array_repr(str_array):
    assert repr(str_array) == "H5Array(['a', 'bc', 'd', 'efg', 'h'], shape=(5,), dtype='<U3')"


def test_setitem_str(str_array):
    str_array[[0, 1]] = np.array(["A", "BBBB"])
    assert np.array_equal(str_array, ["A", "BBBB", "d", "efg", "h"])
    assert str_array.dtype == np.dtype("<U4")


def test_array_str_type_casting(str_array, array):
    assert str_array.astype(str).dtype == np.dtype("<U3")
    assert array.astype(str).dtype == np.dtype("<U32")


@pytest.mark.xfail
def test_array_str_cast_int_should_fail(str_array):
    _ = str_array.astype(int)[:]


def test_iter_chunks_str_array(str_array):
    _, chunk = list(str_array.iter_chunks())[0]
    assert np.issubdtype(chunk.dtype, str)


def test_str_array_should_convert_to_numpy(str_array):
    assert type(np.array(str_array)) == np.ndarray
    assert type(np.array(str_array[0:1])) == np.ndarray
