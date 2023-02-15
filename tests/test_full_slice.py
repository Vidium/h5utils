# coding: utf-8

# ====================================================
# imports
from ch5mpy.h5array.slice import FullSlice


# ====================================================
# code
def test_true_stop_step1():
    f = FullSlice(0, 3, 1, 10)
    assert f.true_stop == 2


def test_true_stop_step3():
    f = FullSlice(5, 13, 3, 20)
    assert f.true_stop == 11


def test_true_stop_no_len():
    f = FullSlice(0, 0, 1, 0)
    assert f.true_stop == 0


def test_length():
    f = FullSlice(0, 3, 1, 10)
    assert len(f) == 3
