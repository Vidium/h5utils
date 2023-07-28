import ch5mpy as ch
from ch5mpy.h5array.chunks.repeated_array import RepeatedArray


def test_subset():
    arr = ch.arange_nd((8, 1, 6, 1))
    r_arr = RepeatedArray(arr, (8, 7, 6, 5))

    sub = r_arr[0:2, 3:7, 1:4]
    assert sub.shape == (2, 4, 3, 5)


def test_subset_larger_shape():
    arr = ch.arange_nd((4, 5))
    r_arr = RepeatedArray(arr, (3, 4, 5))

    sub = r_arr[0:1]
    assert sub.shape == (1, 4, 5)
