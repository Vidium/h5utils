# coding: utf-8

# ====================================================
# imports
from ch5mpy.h5array.inplace import get_chunks
from ch5mpy.h5array.slice import FullSlice


# ====================================================
# code
def test_1d_smaller_than_nb_elements():
    assert get_chunks(10, (5,), 1) == ((FullSlice.whole_axis(5),),)


def test_1d_greater_than_nb_elements():
    assert get_chunks(10, (15,), 1) == ((FullSlice(0, 10, 1, 15),),
                                        (FullSlice(10, 15, 1, 15),))


def test_1d_greater_than_nb_elements_multiple():
    assert get_chunks(10, (30,), 1) == (
        (FullSlice(0, 10, 1, 30),),
        (FullSlice(10, 20, 1, 30),),
        (FullSlice(20, 30, 1, 30),),
    )


def test_2d_array_smaller_than_nb_elements():
    assert get_chunks(100, (2, 10), 1) == (
        (FullSlice.whole_axis(2),),
    )


def test_2d_array_1row():
    assert get_chunks(10, (8, 10), 1) == (
        (FullSlice.one(0, 8), FullSlice.whole_axis(10),),
        (FullSlice.one(1, 8), FullSlice.whole_axis(10),),
        (FullSlice.one(2, 8), FullSlice.whole_axis(10),),
        (FullSlice.one(3, 8), FullSlice.whole_axis(10),),
        (FullSlice.one(4, 8), FullSlice.whole_axis(10),),
        (FullSlice.one(5, 8), FullSlice.whole_axis(10),),
        (FullSlice.one(6, 8), FullSlice.whole_axis(10),),
        (FullSlice.one(7, 8), FullSlice.whole_axis(10),),
    )


def test_2d_array_2rows():
    assert get_chunks(20, (8, 10), 1) == (
        (FullSlice(0, 2, 1, 8), FullSlice.whole_axis(10),),
        (FullSlice(2, 4, 1, 8), FullSlice.whole_axis(10),),
        (FullSlice(4, 6, 1, 8), FullSlice.whole_axis(10),),
        (FullSlice(6, 8, 1, 8), FullSlice.whole_axis(10),),
    )


def test_2d_array_0rows():
    assert get_chunks(6, (8, 10), 1) == (
        (FullSlice.one(0, 8), FullSlice(0, 6, 1, 10)),
        (FullSlice.one(0, 8), FullSlice(6, 10, 1, 10)),
        (FullSlice.one(1, 8), FullSlice(0, 6, 1, 10)),
        (FullSlice.one(1, 8), FullSlice(6, 10, 1, 10)),
        (FullSlice.one(2, 8), FullSlice(0, 6, 1, 10)),
        (FullSlice.one(2, 8), FullSlice(6, 10, 1, 10)),
        (FullSlice.one(3, 8), FullSlice(0, 6, 1, 10)),
        (FullSlice.one(3, 8), FullSlice(6, 10, 1, 10)),
        (FullSlice.one(4, 8), FullSlice(0, 6, 1, 10)),
        (FullSlice.one(4, 8), FullSlice(6, 10, 1, 10)),
        (FullSlice.one(5, 8), FullSlice(0, 6, 1, 10)),
        (FullSlice.one(5, 8), FullSlice(6, 10, 1, 10)),
        (FullSlice.one(6, 8), FullSlice(0, 6, 1, 10)),
        (FullSlice.one(6, 8), FullSlice(6, 10, 1, 10)),
        (FullSlice.one(7, 8), FullSlice(0, 6, 1, 10)),
        (FullSlice.one(7, 8), FullSlice(6, 10, 1, 10)),
    )


def test_3d_array_smaller_than_nb_elements():
    assert get_chunks(200, (5, 5, 5), 1) == (
        (FullSlice.whole_axis(5),),
    )


def test_3d_array_1_array():
    assert get_chunks(30, (5, 5, 5), 1) == (
        (FullSlice.one(0, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice.one(1, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice.one(2, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice.one(3, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice.one(4, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
    )


def test_3d_array_2_arrays():
    assert get_chunks(60, (5, 5, 5), 1) == (
        (FullSlice(0, 2, 1, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice(2, 4, 1, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
    )


def test_3d_array_2rows():
    assert get_chunks(20, (5, 5, 5), 1) == (
        (FullSlice.one(0, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(0, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(1, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(1, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(2, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(2, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(3, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(3, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(4, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(4, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
    )


def test_3d_array_1row():
    assert get_chunks(6, (5, 5, 5), 1) == (
        (FullSlice.one(0, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(0, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(0, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(0, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(0, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(1, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(1, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(1, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(1, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(1, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(2, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(2, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(2, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(2, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(2, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(3, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(3, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(3, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(3, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(3, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(4, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(4, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(4, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(4, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (FullSlice.one(4, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
    )


def test_3d_array_0rows():
    assert get_chunks(3, (5, 5, 5), 1) == (
        (FullSlice.one(0, 5), FullSlice.one(0, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(0, 5), FullSlice.one(0, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(0, 5), FullSlice.one(1, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(0, 5), FullSlice.one(1, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(0, 5), FullSlice.one(2, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(0, 5), FullSlice.one(2, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(0, 5), FullSlice.one(3, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(0, 5), FullSlice.one(3, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(0, 5), FullSlice.one(4, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(0, 5), FullSlice.one(4, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(0, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(0, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(1, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(1, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(2, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(2, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(3, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(3, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(4, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(1, 5), FullSlice.one(4, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(0, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(0, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(1, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(1, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(2, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(2, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(3, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(3, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(4, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(2, 5), FullSlice.one(4, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(0, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(0, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(1, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(1, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(2, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(2, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(3, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(3, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(4, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(3, 5), FullSlice.one(4, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(0, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(0, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(1, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(1, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(2, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(2, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(3, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(3, 5), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(4, 5), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(4, 5), FullSlice.one(4, 5), FullSlice(3, 5, 1, 5)),
    )


def test_3d_array():
    assert get_chunks(24, (3, 4, 5), 8) == (
        (FullSlice.one(0, 3), FullSlice.one(0, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(0, 3), FullSlice.one(0, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(0, 3), FullSlice.one(1, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(0, 3), FullSlice.one(1, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(0, 3), FullSlice.one(2, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(0, 3), FullSlice.one(2, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(0, 3), FullSlice.one(3, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(0, 3), FullSlice.one(3, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(1, 3), FullSlice.one(0, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(1, 3), FullSlice.one(0, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(1, 3), FullSlice.one(1, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(1, 3), FullSlice.one(1, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(1, 3), FullSlice.one(2, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(1, 3), FullSlice.one(2, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(1, 3), FullSlice.one(3, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(1, 3), FullSlice.one(3, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(2, 3), FullSlice.one(0, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(2, 3), FullSlice.one(0, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(2, 3), FullSlice.one(1, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(2, 3), FullSlice.one(1, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(2, 3), FullSlice.one(2, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(2, 3), FullSlice.one(2, 4), FullSlice(3, 5, 1, 5)),
        (FullSlice.one(2, 3), FullSlice.one(3, 4), FullSlice(0, 3, 1, 5)),
        (FullSlice.one(2, 3), FullSlice.one(3, 4), FullSlice(3, 5, 1, 5)),
    )
