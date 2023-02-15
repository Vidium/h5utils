# coding: utf-8

# ====================================================
# imports
import pytest
import numpy as np
from pathlib import Path

from ch5mpy import File
from ch5mpy import H5Dict
from ch5mpy import H5Mode
from ch5mpy import H5Array
from ch5mpy import write_object


# ====================================================
# code
@pytest.fixture
def backed_dict() -> H5Dict:
    data = {
        "a": 1,
        "b": [1, 2, 3],
        "c": {"d": "test", "e": np.arange(100)},
        "f": np.zeros((10, 10, 10)),
    }

    with File("backed_dict", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "uns", data)

    yield H5Dict(File("backed_dict", H5Mode.READ_WRITE)["uns"])

    Path("backed_dict").unlink()


def test_backed_dict_creation(backed_dict):
    assert isinstance(backed_dict, H5Dict)


def test_backed_dict_can_iterate_through_keys(backed_dict):
    assert list(iter(backed_dict)) == ["a", "b", "c", "f"]


def test_backed_dict_has_correct_keys(backed_dict):
    assert list(backed_dict.keys()) == ["a", "b", "c", "f"]


def test_backed_dict_can_get_regular_values(backed_dict):
    assert backed_dict["a"] == 1


def test_backed_dict_can_get_string(backed_dict):
    assert backed_dict["c"]["d"] == "test"


def test_backed_dict_can_get_with_getattr(backed_dict):
    assert isinstance(backed_dict.c.e, H5Array)


def test_backed_dict_should_return_string(backed_dict):
    assert isinstance(backed_dict["c"]["d"], str)


def test_backed_dict_gets_nested_backed_dicts(backed_dict):
    assert isinstance(backed_dict["c"], H5Dict)


def test_backed_dict_has_correct_values(backed_dict):
    values_list = list(backed_dict.values())

    assert (
        values_list[0] == 1
        and np.array_equal(values_list[1], [1, 2, 3])
        and list(values_list[2].keys()) == ["d", "e"]
    )


def test_backed_dict_has_correct_items(backed_dict):
    assert list(backed_dict.items())[0] == ("a", 1)


def test_backed_dict_can_set_regular_value(backed_dict):
    backed_dict["a"] = 5

    assert np.all(backed_dict["a"] == 5)


def test_backed_dict_can_set_array_value(backed_dict):
    backed_dict["b"][1] = 6

    assert np.all(backed_dict["b"] == [1, 6, 3])


def test_backed_dict_can_set_new_regular_value(backed_dict):
    backed_dict["x"] = 9

    assert backed_dict["x"] == 9


def test_backed_dict_can_set_new_array(backed_dict):
    backed_dict["y"] = np.array([1, 2, 3])

    assert np.all(backed_dict["y"] == [1, 2, 3])


def test_backed_dict_can_set_new_dict(backed_dict):
    backed_dict["z"] = {"l": 10, "m": [10, 11, 12], "n": {"o": 13}}

    assert (
        isinstance(backed_dict["z"], H5Dict)
        and backed_dict["z"]["l"] == 10
        and np.all(backed_dict["z"]["m"] == [10, 11, 12])
        and isinstance(backed_dict["z"]["n"], H5Dict)
        and backed_dict["z"]["n"]["o"] == 13
    )


def test_backed_dict_can_delete_regular_value(backed_dict):
    del backed_dict["a"]

    assert "a" not in backed_dict.keys()


def test_backed_dict_can_delete_array(backed_dict):
    del backed_dict["b"]

    assert "b" not in backed_dict.keys()


def test_backed_dict_can_delete_dict(backed_dict):
    del backed_dict["c"]

    assert "c" not in backed_dict.keys()


def test_backed_dict_can_close_file(backed_dict):
    backed_dict.close()

    with pytest.raises(ValueError):
        _ = backed_dict["x"]


def test_backed_dict_copy_should_be_regular_dict(backed_dict):
    c = backed_dict.copy()

    assert isinstance(c, dict)


def test_backed_dict_copy_should_have_same_keys(backed_dict):
    c = backed_dict.copy()

    assert c.keys() == backed_dict.keys()


def test_backed_dict_copy_nested_backed_dict_should_be_dict(backed_dict):
    c = backed_dict.copy()

    assert isinstance(c["c"], dict)


def test_backed_dict_copy_dataset_proxy_should_be_array(backed_dict):
    c = backed_dict.copy()

    assert isinstance(c["b"], H5Array)
