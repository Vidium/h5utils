import numpy as np

import ch5mpy as ch


def test_ones(group):
    arr = ch.ones(5, "test", group)

    assert isinstance(arr, ch.H5Array)
    assert arr.shape == (5,)
    assert np.array_equal(arr, np.ones(5))


def test_ones_set_in_H5Dict(group):
    d = ch.H5Dict(group)

    d["test_set"] = ch.ones.anonymous((5, 5))

    assert np.array_equal(d["test_set"], np.ones((5, 5)))


def test_ones_set_in_H5Dict_nested(group):
    d = ch.H5Dict(group)

    d["test_set_nested"] = {"some_a": {"some_b": ch.ones.anonymous((5, 5))}}

    assert np.array_equal(d["test_set_nested"]["some_a"]["some_b"], np.ones((5, 5)))


def test_rand(group):
    np.random.seed(42)
    arr = ch.random.rand(2, 3, name="test", loc=group)

    assert isinstance(arr, ch.H5Array)
    assert np.array_equal(
        arr,
        np.array(
            [
                [0.3745401188473625, 0.9507143064099162, 0.7319939418114051],
                [0.5986584841970366, 0.15601864044243652, 0.15599452033620265],
            ]
        ),
    )
