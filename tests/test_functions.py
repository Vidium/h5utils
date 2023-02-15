# coding: utf-8

# ====================================================
# imports
import numpy as np


# ====================================================
# code
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


def test_sum_where(small_large_array):
    assert np.array_equal(np.sum(small_large_array, axis=1, where=[True, False, False, True, True]),
                          np.array([[30., 0., 0., 42., 46.],
                                    [110., 0., 0., 122., 126.],
                                    [190., 0., 0., 202., 206.]]))


def test_sum_where_multiple_rows(small_large_array):
    small_large_array.MAX_MEM_USAGE = str(40 * small_large_array.dtype.itemsize)
    assert np.array_equal(np.sum(small_large_array, axis=1, where=[True, False, False, True, True]),
                          np.array([[30., 0., 0., 42., 46.],
                                    [110., 0., 0., 122., 126.],
                                    [190., 0., 0., 202., 206.]]))


def test_sum_where_few_elements(small_large_array):
    small_large_array.MAX_MEM_USAGE = str(3 * small_large_array.dtype.itemsize)
    assert np.array_equal(np.sum(small_large_array, axis=1, where=[True, False, False, True, True]),
                          np.array([[30., 0., 0., 42., 46.],
                                    [110., 0., 0., 122., 126.],
                                    [190., 0., 0., 202., 206.]]))


def test_all(array):
    assert not np.all(array)


def test_all_axis_keepdims(array):
    assert np.array_equal(np.all(array, axis=1, keepdims=True), np.array([[False], [True], [True], [True], [True],
                                                                          [True],  [True], [True], [True], [True]]))


def test_floor(array):
    array += 0.7
    assert np.array_equal(np.floor(array), np.arange(100.).reshape((10, 10)))


def test_ceil(array):
    array += 0.2
    assert np.array_equal(np.ceil(array), np.arange(1., 101.).reshape((10, 10)))


# def test_ceil_large(large_array):
#     np.ceil(large_array)


def test_trunc(array):
    array += 0.2
    assert np.array_equal(np.trunc(array), np.arange(100.).reshape((10, 10)))


def test_prod(array):
    assert np.array_equal(
        np.prod(array, axis=0, where=[True, False, False, False, False, True, True, True, True, True]),
        [0, 1, 1, 1, 1, 6393838623046875, 9585618768101376, 13865696119905399, 19511273389031424, 26853950884211451]
    )


def test_exp(array):
    assert np.array_equal(
        np.exp(array),
        np.exp(np.arange(100.).reshape((10, 10)))
    )


def test_exp_where(small_array):
    out = np.array([1., 1., 1., 1., 1.])
    np.exp(small_array, out, where=[True, False, False, True, True])

    assert np.array_equal(out, [np.exp(1), 1., 1., np.exp(4), np.exp(5)])


def test_expm1(small_array):
    assert np.array_equal(np.expm1(small_array), np.expm1([1, 2, 3, 4, 5]))


def test_isfinite(small_array):
    small_array[0] = np.inf
    assert np.array_equal(np.isfinite(small_array), [False, True, True, True, True])


def test_equal(small_array):
    assert np.all(np.equal(small_array, [1, 2, 3, 4, 5]))
    assert np.array_equal(np.equal(small_array, 2), [False, True, False, False, False])


def test_equal_where(small_array):
    assert np.array_equal(np.equal(small_array, [1, 2, 3, 3, 3], where=[True, True, False, False, True]),
                          [True, True, False, False, False])


def test_greater_element(small_array):
    assert np.array_equal(np.greater(small_array, 3), [False, False, False, True, True])


def test_multiply(small_array):
    assert np.array_equal(np.multiply(small_array, [2, 3, 4, 5, 6]), [2, 6, 12, 20, 30])


def test_multiply_where(small_array):
    assert np.array_equal(np.multiply(small_array, [2, 3, 4, 5, 6], where=[True, False, False, True, True]),
                          [2, 2, 3, 20, 30])


def test_isinf(small_array):
    small_array[1] = np.inf
    assert np.array_equal(np.isinf(small_array), [False, True, False, False, False])


def test_isinf_out(small_array):
    small_array[3] = np.inf
    out = np.array([1, 2, 3, 4, 5])
    np.isinf(small_array, where=[True, False, False, True, True], out=out)
    assert np.array_equal(out, [False, 2, 3, True, False])


def test_logical_and(small_array):
    assert np.array_equal(np.logical_and(small_array, [True, True, False, False, True]),
                          [True, True, False, False, True])
