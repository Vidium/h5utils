import numpy as np

import ch5mpy as ch


def test_ones(group):
    arr = ch.ones(5, "test", group)

    assert isinstance(arr, ch.H5Array)
    assert arr.shape == (5,)
    assert np.array_equal(arr, np.ones(5))


def test_ones_set_in_H5Dict(group):
    d = ch.H5Dict(group)

    d["test_set"] = ch.ones.p((5, 5))

    assert np.array_equal(d["test_set"], np.ones((5, 5)))
