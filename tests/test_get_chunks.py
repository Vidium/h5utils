from ch5mpy.h5array.chunks.iter import _get_chunk_indices
from ch5mpy.indexing.slice import FullSlice, SingleIndex


def test_1d_smaller_than_nb_elements():
    assert _get_chunk_indices(10, (5,)) == ((FullSlice.whole_axis(5),),)


def test_1d_greater_than_nb_elements():
    assert _get_chunk_indices(10, (15,)) == ((FullSlice(0, 10, 1, 15),), (FullSlice(10, 15, 1, 15),))


def test_1d_greater_than_nb_elements_multiple():
    assert _get_chunk_indices(10, (30,)) == (
        (FullSlice(0, 10, 1, 30),),
        (FullSlice(10, 20, 1, 30),),
        (FullSlice(20, 30, 1, 30),),
    )


def test_2d_array_smaller_than_nb_elements():
    assert _get_chunk_indices(100, (2, 10)) == ((FullSlice.whole_axis(2), FullSlice.whole_axis(10)),)


def test_2d_array_1row():
    assert _get_chunk_indices(10, (8, 10)) == (
        (
            FullSlice.one(0, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice.one(1, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice.one(2, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice.one(3, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice.one(4, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice.one(5, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice.one(6, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice.one(7, 8),
            FullSlice.whole_axis(10),
        ),
    )


def test_2d_array_2rows():
    assert _get_chunk_indices(20, (8, 10)) == (
        (
            FullSlice(0, 2, 1, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice(2, 4, 1, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice(4, 6, 1, 8),
            FullSlice.whole_axis(10),
        ),
        (
            FullSlice(6, 8, 1, 8),
            FullSlice.whole_axis(10),
        ),
    )


def test_2d_array_0rows():
    assert _get_chunk_indices(6, (8, 10)) == (
        (SingleIndex(0, 8), FullSlice(0, 6, 1, 10)),
        (SingleIndex(0, 8), FullSlice(6, 10, 1, 10)),
        (SingleIndex(1, 8), FullSlice(0, 6, 1, 10)),
        (SingleIndex(1, 8), FullSlice(6, 10, 1, 10)),
        (SingleIndex(2, 8), FullSlice(0, 6, 1, 10)),
        (SingleIndex(2, 8), FullSlice(6, 10, 1, 10)),
        (SingleIndex(3, 8), FullSlice(0, 6, 1, 10)),
        (SingleIndex(3, 8), FullSlice(6, 10, 1, 10)),
        (SingleIndex(4, 8), FullSlice(0, 6, 1, 10)),
        (SingleIndex(4, 8), FullSlice(6, 10, 1, 10)),
        (SingleIndex(5, 8), FullSlice(0, 6, 1, 10)),
        (SingleIndex(5, 8), FullSlice(6, 10, 1, 10)),
        (SingleIndex(6, 8), FullSlice(0, 6, 1, 10)),
        (SingleIndex(6, 8), FullSlice(6, 10, 1, 10)),
        (SingleIndex(7, 8), FullSlice(0, 6, 1, 10)),
        (SingleIndex(7, 8), FullSlice(6, 10, 1, 10)),
    )


def test_3d_array_smaller_than_nb_elements():
    assert _get_chunk_indices(200, (5, 5, 5)) == (
        (FullSlice.whole_axis(5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
    )


def test_3d_array_1_array():
    assert _get_chunk_indices(30, (5, 5, 5)) == (
        (FullSlice.one(0, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice.one(1, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice.one(2, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice.one(3, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice.one(4, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
    )


def test_3d_array_2_arrays():
    assert _get_chunk_indices(60, (5, 5, 5)) == (
        (FullSlice(0, 2, 1, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice(2, 4, 1, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
        (FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5), FullSlice.whole_axis(5)),
    )


def test_3d_array_2rows():
    assert _get_chunk_indices(20, (5, 5, 5)) == (
        (SingleIndex(0, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(0, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(1, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(1, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(2, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(2, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(3, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(3, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(4, 5), FullSlice(0, 4, 1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(4, 5), FullSlice(4, 5, 1, 5), FullSlice.whole_axis(5)),
    )


def test_3d_array_1row():
    assert _get_chunk_indices(6, (5, 5, 5)) == (
        (SingleIndex(0, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (SingleIndex(0, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(0, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (SingleIndex(0, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (SingleIndex(0, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
        (SingleIndex(1, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (SingleIndex(1, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(1, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (SingleIndex(1, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (SingleIndex(1, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
        (SingleIndex(2, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (SingleIndex(2, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(2, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (SingleIndex(2, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (SingleIndex(2, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
        (SingleIndex(3, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (SingleIndex(3, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(3, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (SingleIndex(3, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (SingleIndex(3, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
        (SingleIndex(4, 5), FullSlice.one(0, 5), FullSlice.whole_axis(5)),
        (SingleIndex(4, 5), FullSlice.one(1, 5), FullSlice.whole_axis(5)),
        (SingleIndex(4, 5), FullSlice.one(2, 5), FullSlice.whole_axis(5)),
        (SingleIndex(4, 5), FullSlice.one(3, 5), FullSlice.whole_axis(5)),
        (SingleIndex(4, 5), FullSlice.one(4, 5), FullSlice.whole_axis(5)),
    )


def test_3d_array_0rows():
    assert _get_chunk_indices(3, (5, 5, 5)) == (
        (SingleIndex(0, 5), SingleIndex(0, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(0, 5), SingleIndex(0, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(0, 5), SingleIndex(1, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(0, 5), SingleIndex(1, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(0, 5), SingleIndex(2, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(0, 5), SingleIndex(2, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(0, 5), SingleIndex(3, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(0, 5), SingleIndex(3, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(0, 5), SingleIndex(4, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(0, 5), SingleIndex(4, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(0, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(0, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(1, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(1, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(2, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(2, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(3, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(3, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(4, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(1, 5), SingleIndex(4, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(0, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(0, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(1, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(1, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(2, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(2, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(3, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(3, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(4, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(2, 5), SingleIndex(4, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(0, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(0, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(1, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(1, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(2, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(2, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(3, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(3, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(4, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(3, 5), SingleIndex(4, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(0, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(0, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(1, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(1, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(2, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(2, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(3, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(3, 5), FullSlice(3, 5, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(4, 5), FullSlice(0, 3, 1, 5)),
        (SingleIndex(4, 5), SingleIndex(4, 5), FullSlice(3, 5, 1, 5)),
    )


def test_3d_array():
    assert _get_chunk_indices(3, (3, 4, 5)) == (
        (SingleIndex(0, 3), SingleIndex(0, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(0, 3), SingleIndex(0, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(0, 3), SingleIndex(1, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(0, 3), SingleIndex(1, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(0, 3), SingleIndex(2, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(0, 3), SingleIndex(2, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(0, 3), SingleIndex(3, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(0, 3), SingleIndex(3, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(1, 3), SingleIndex(0, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(1, 3), SingleIndex(0, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(1, 3), SingleIndex(1, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(1, 3), SingleIndex(1, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(1, 3), SingleIndex(2, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(1, 3), SingleIndex(2, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(1, 3), SingleIndex(3, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(1, 3), SingleIndex(3, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(2, 3), SingleIndex(0, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(2, 3), SingleIndex(0, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(2, 3), SingleIndex(1, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(2, 3), SingleIndex(1, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(2, 3), SingleIndex(2, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(2, 3), SingleIndex(2, 4), FullSlice(3, 5, 1, 5)),
        (SingleIndex(2, 3), SingleIndex(3, 4), FullSlice(0, 3, 1, 5)),
        (SingleIndex(2, 3), SingleIndex(3, 4), FullSlice(3, 5, 1, 5)),
    )
