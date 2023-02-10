# coding: utf-8

# ====================================================
# imports
import numpy as np


# ====================================================
# code
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
    small_large_array.MAX_MEM_USAGE = str(3 * small_large_array.dtype.itemsize)
    assert np.array_equal(np.sum(small_large_array, axis=0), np.array([[60, 63, 66, 69, 72],
                                                                       [75, 78, 81, 84, 87],
                                                                       [90, 93, 96, 99, 102],
                                                                       [105, 108, 111, 114, 117]]))


def test_sum_axis_2(small_large_array):
    small_large_array.MAX_MEM_USAGE = str(3 * small_large_array.dtype.itemsize)
    assert np.array_equal(np.sum(small_large_array, axis=2), np.array([[10., 35., 60., 85.],
                                                                       [110., 135., 160., 185.],
                                                                       [210., 235., 260., 285.]]))
