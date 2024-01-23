import pickle
import numpy as np

from ch5mpy import write_object


def test_should_pickle_dataset(group):
    write_object([1, 2, 3], group, "something")

    pickled_obj = pickle.dumps(group["something"], protocol=pickle.HIGHEST_PROTOCOL)
    unpickled_obj = pickle.loads(pickled_obj)

    assert np.array_equal(unpickled_obj[:], [1, 2, 3])


def test_should_pickle_dict(group):
    write_object({"a": 1, "b": "2"}, group, "something")

    pickled_obj = pickle.dumps(group["something"], protocol=pickle.HIGHEST_PROTOCOL)
    unpickled_obj = pickle.loads(pickled_obj)

    assert unpickled_obj["a"][()] == 1
    assert unpickled_obj["b"][()] == b"2"
