# coding: utf-8

# ====================================================
# imports
import pickle
import numpy as np

from ch5mpy.write import write_object


# ====================================================
# code
def test_should_pickle_dataset(group):
    write_object(group, "something", [1, 2, 3])

    pickled_obj = pickle.dumps(group["something"], protocol=pickle.HIGHEST_PROTOCOL)
    unpickled_obj = pickle.loads(pickled_obj)

    assert np.array_equal(unpickled_obj[:], [1, 2, 3])


def test_should_pickle_dict(group):
    write_object(group, "something", {"a": 1, "b": "2"})

    pickled_obj = pickle.dumps(group["something"], protocol=pickle.HIGHEST_PROTOCOL)
    unpickled_obj = pickle.loads(pickled_obj)

    assert unpickled_obj["a"][()] == 1
    assert unpickled_obj["b"][()] == b"2"
