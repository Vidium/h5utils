import numpy as np


def test_should_add_inplace(chunked_array):
    chunked_array += 1
    assert np.array_equal(chunked_array, np.arange(100).reshape((10, 10)) + 1)


def test_should_resize_larger(chunked_array):
    chunked_array.expand(2)

    assert chunked_array.shape == (12, 10)


def test_should_resize_smaller(chunked_array):
    chunked_array.contract(1, 1)

    assert chunked_array.shape == (10, 9)
