# coding: utf-8

# ====================================================
# imports
from h5utils.h5array.inplace import get_chunks


# ====================================================
# code
def test_1d_smaller_than_nb_elements():
    assert get_chunks(10, (5,), 1) == ((slice(None),),)


def test_1d_greater_than_nb_elements():
    assert get_chunks(10, (15,), 1) == ((slice(0, 10),), (slice(10, 15),))


def test_1d_greater_than_nb_elements_multiple():
    assert get_chunks(10, (30,), 1) == (
        (slice(0, 10),),
        (slice(10, 20),),
        (slice(20, 30),),
    )


def test_2d_array_smaller_than_nb_elements():
    assert get_chunks(100, (2, 10), 1) == (
        (slice(None,),),
    )


def test_2d_array_1row():
    assert get_chunks(10, (8, 10), 1) == (
        (0, slice(None,),),
        (1, slice(None,),),
        (2, slice(None,),),
        (3, slice(None,),),
        (4, slice(None,),),
        (5, slice(None,),),
        (6, slice(None,),),
        (7, slice(None,),),
    )


def test_2d_array_2rows():
    assert get_chunks(20, (8, 10), 1) == (
        (slice(0, 2), slice(None,),),
        (slice(2, 4), slice(None,),),
        (slice(4, 6), slice(None,),),
        (slice(6, 8), slice(None,),),
    )


def test_2d_array_0rows():
    assert get_chunks(6, (8, 10), 1) == (
        (0, slice(0, 6)),
        (0, slice(6, 10)),
        (1, slice(0, 6)),
        (1, slice(6, 10)),
        (2, slice(0, 6)),
        (2, slice(6, 10)),
        (3, slice(0, 6)),
        (3, slice(6, 10)),
        (4, slice(0, 6)),
        (4, slice(6, 10)),
        (5, slice(0, 6)),
        (5, slice(6, 10)),
        (6, slice(0, 6)),
        (6, slice(6, 10)),
        (7, slice(0, 6)),
        (7, slice(6, 10)),
    )


def test_3d_array_smaller_than_nb_elements():
    assert get_chunks(200, (5, 5, 5), 1) == (
        (slice(None,),),
    )


def test_3d_array_1_array():
    assert get_chunks(30, (5, 5, 5), 1) == (
        (0, slice(None, None, None), slice(None, None, None)),
        (1, slice(None, None, None), slice(None, None, None)),
        (2, slice(None, None, None), slice(None, None, None)),
        (3, slice(None, None, None), slice(None, None, None)),
        (4, slice(None, None, None), slice(None, None, None)),
    )


def test_3d_array_2_arrays():
    assert get_chunks(60, (5, 5, 5), 1) == (
        (slice(0, 2, None), slice(None, None, None), slice(None, None, None)),
        (slice(2, 4, None), slice(None, None, None), slice(None, None, None)),
        (slice(4, 5, None), slice(None, None, None), slice(None, None, None)),
    )


def test_3d_array_2rows():
    assert get_chunks(20, (5, 5, 5), 1) == (
        (0, slice(0, 4, None), slice(None, None, None)),
        (0, slice(4, 5, None), slice(None, None, None)),
        (1, slice(0, 4, None), slice(None, None, None)),
        (1, slice(4, 5, None), slice(None, None, None)),
        (2, slice(0, 4, None), slice(None, None, None)),
        (2, slice(4, 5, None), slice(None, None, None)),
        (3, slice(0, 4, None), slice(None, None, None)),
        (3, slice(4, 5, None), slice(None, None, None)),
        (4, slice(0, 4, None), slice(None, None, None)),
        (4, slice(4, 5, None), slice(None, None, None)),
    )


def test_3d_array_1row():
    assert get_chunks(6, (5, 5, 5), 1) == (
        (0, 0, slice(None, None, None)),
        (0, 1, slice(None, None, None)),
        (0, 2, slice(None, None, None)),
        (0, 3, slice(None, None, None)),
        (0, 4, slice(None, None, None)),
        (1, 0, slice(None, None, None)),
        (1, 1, slice(None, None, None)),
        (1, 2, slice(None, None, None)),
        (1, 3, slice(None, None, None)),
        (1, 4, slice(None, None, None)),
        (2, 0, slice(None, None, None)),
        (2, 1, slice(None, None, None)),
        (2, 2, slice(None, None, None)),
        (2, 3, slice(None, None, None)),
        (2, 4, slice(None, None, None)),
        (3, 0, slice(None, None, None)),
        (3, 1, slice(None, None, None)),
        (3, 2, slice(None, None, None)),
        (3, 3, slice(None, None, None)),
        (3, 4, slice(None, None, None)),
        (4, 0, slice(None, None, None)),
        (4, 1, slice(None, None, None)),
        (4, 2, slice(None, None, None)),
        (4, 3, slice(None, None, None)),
        (4, 4, slice(None, None, None)),
    )


def test_3d_array_0rows():
    assert get_chunks(3, (5, 5, 5), 1) == (
        (0, 0, slice(0, 3, None)),
        (0, 0, slice(3, 5, None)),
        (0, 1, slice(0, 3, None)),
        (0, 1, slice(3, 5, None)),
        (0, 2, slice(0, 3, None)),
        (0, 2, slice(3, 5, None)),
        (0, 3, slice(0, 3, None)),
        (0, 3, slice(3, 5, None)),
        (0, 4, slice(0, 3, None)),
        (0, 4, slice(3, 5, None)),
        (1, 0, slice(0, 3, None)),
        (1, 0, slice(3, 5, None)),
        (1, 1, slice(0, 3, None)),
        (1, 1, slice(3, 5, None)),
        (1, 2, slice(0, 3, None)),
        (1, 2, slice(3, 5, None)),
        (1, 3, slice(0, 3, None)),
        (1, 3, slice(3, 5, None)),
        (1, 4, slice(0, 3, None)),
        (1, 4, slice(3, 5, None)),
        (2, 0, slice(0, 3, None)),
        (2, 0, slice(3, 5, None)),
        (2, 1, slice(0, 3, None)),
        (2, 1, slice(3, 5, None)),
        (2, 2, slice(0, 3, None)),
        (2, 2, slice(3, 5, None)),
        (2, 3, slice(0, 3, None)),
        (2, 3, slice(3, 5, None)),
        (2, 4, slice(0, 3, None)),
        (2, 4, slice(3, 5, None)),
        (3, 0, slice(0, 3, None)),
        (3, 0, slice(3, 5, None)),
        (3, 1, slice(0, 3, None)),
        (3, 1, slice(3, 5, None)),
        (3, 2, slice(0, 3, None)),
        (3, 2, slice(3, 5, None)),
        (3, 3, slice(0, 3, None)),
        (3, 3, slice(3, 5, None)),
        (3, 4, slice(0, 3, None)),
        (3, 4, slice(3, 5, None)),
        (4, 0, slice(0, 3, None)),
        (4, 0, slice(3, 5, None)),
        (4, 1, slice(0, 3, None)),
        (4, 1, slice(3, 5, None)),
        (4, 2, slice(0, 3, None)),
        (4, 2, slice(3, 5, None)),
        (4, 3, slice(0, 3, None)),
        (4, 3, slice(3, 5, None)),
        (4, 4, slice(0, 3, None)),
        (4, 4, slice(3, 5, None)),
    )


def test_3d_array():
    assert get_chunks(24, (3, 4, 5), 8) == (
        (0, 0, slice(0, 3, None)),
        (0, 0, slice(3, 5, None)),
        (0, 1, slice(0, 3, None)),
        (0, 1, slice(3, 5, None)),
        (0, 2, slice(0, 3, None)),
        (0, 2, slice(3, 5, None)),
        (0, 3, slice(0, 3, None)),
        (0, 3, slice(3, 5, None)),
        (1, 0, slice(0, 3, None)),
        (1, 0, slice(3, 5, None)),
        (1, 1, slice(0, 3, None)),
        (1, 1, slice(3, 5, None)),
        (1, 2, slice(0, 3, None)),
        (1, 2, slice(3, 5, None)),
        (1, 3, slice(0, 3, None)),
        (1, 3, slice(3, 5, None)),
        (2, 0, slice(0, 3, None)),
        (2, 0, slice(3, 5, None)),
        (2, 1, slice(0, 3, None)),
        (2, 1, slice(3, 5, None)),
        (2, 2, slice(0, 3, None)),
        (2, 2, slice(3, 5, None)),
        (2, 3, slice(0, 3, None)),
        (2, 3, slice(3, 5, None)),
    )
