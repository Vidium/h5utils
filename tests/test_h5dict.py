from __future__ import annotations

import numpy as np
import pytest

from ch5mpy import H5Array, H5Dict


def test_h5_dict_creation(h5_dict):
    assert isinstance(h5_dict, H5Dict)


def test_h5_dict_can_iterate_through_keys(h5_dict):
    assert list(iter(h5_dict)) == ["a", "b", "c", "f"]


def test_h5_dict_has_correct_keys(h5_dict):
    assert list(h5_dict.keys()) == ["a", "b", "c", "f"]


def test_h5_dict_can_get_regular_values(h5_dict):
    assert h5_dict["a"] == 1


def test_h5_dict_can_get_string(h5_dict):
    assert h5_dict["c"]["d"] == "test"


def test_h5_dict_can_get_with_getattr(h5_dict):
    assert isinstance(h5_dict.c.e, H5Array)


def test_h5_dict_should_return_string(h5_dict):
    assert isinstance(h5_dict["c"]["d"], str)


def test_h5_dict_gets_nested_h5_dicts(h5_dict):
    assert isinstance(h5_dict["c"], H5Dict)


def test_h5_dict_has_correct_values(h5_dict):
    values_list = list(h5_dict.values())

    assert (
        values_list[0] == 1 and np.array_equal(values_list[1], [1, 2, 3]) and list(values_list[2].keys()) == ["d", "e"]
    )


def test_h5_dict_has_correct_items(h5_dict):
    assert list(h5_dict.items())[0] == ("a", 1)


def test_h5_dict_can_set_regular_value(h5_dict):
    h5_dict["a"] = 5

    assert h5_dict["a"] == 5


def test_h5_dict_can_set_array_value(h5_dict):
    h5_dict["b"][1] = 6

    assert np.all(h5_dict["b"] == [1, 6, 3])


def test_h5_dict_can_set_new_regular_value(h5_dict):
    h5_dict["x"] = 9

    assert h5_dict["x"] == 9


def test_h5_dict_can_set_new_array(h5_dict):
    h5_dict["y"] = np.array([1, 2, 3])

    assert np.all(h5_dict["y"] == [1, 2, 3])


def test_h5_dict_can_set_new_dict(h5_dict):
    h5_dict["z"] = {"l": 10, "m": [10, 11, 12], "n": {"o": 13}}

    assert (
        isinstance(h5_dict["z"], H5Dict)
        and h5_dict["z"]["l"] == 10
        and np.all(h5_dict["z"]["m"] == [10, 11, 12])
        and isinstance(h5_dict["z"]["n"], H5Dict)
        and h5_dict["z"]["n"]["o"] == 13
    )


def test_h5_dict_can_replace_dict(h5_dict):
    h5_dict["c"] = {"d2": "test", "e": np.arange(10, 20)}

    assert isinstance(h5_dict["c"], H5Dict)
    assert "d" not in h5_dict["c"].keys()
    assert h5_dict["c"]["d2"] == "test"
    assert np.array_equal(h5_dict["c"]["e"], np.arange(10, 20))


def test_h5_dict_can_union(h5_dict):
    h5_dict["c"] |= {"d": "new_val", "g": -1}

    assert h5_dict["c"].keys() == {"d", "e", "g"}
    assert h5_dict["c"]["d"] == "new_val"
    assert np.array_equal(h5_dict["c"]["e"], np.arange(100))
    assert h5_dict["c"]["g"] == -1


def test_h5_dict_can_delete_regular_value(h5_dict):
    del h5_dict["a"]

    assert "a" not in h5_dict.keys()


def test_h5_dict_can_delete_array(h5_dict):
    del h5_dict["b"]

    assert "b" not in h5_dict.keys()


def test_h5_dict_can_delete_dict(h5_dict):
    del h5_dict["c"]

    assert "c" not in h5_dict.keys()


def test_h5_dict_can_close_file(h5_dict):
    h5_dict.close()

    with pytest.raises(KeyError):
        _ = h5_dict["a"]


def test_h5_dict_copy_should_be_regular_dict(h5_dict):
    c = h5_dict.copy()

    assert isinstance(c, dict)


def test_h5_dict_copy_should_have_same_keys(h5_dict):
    c = h5_dict.copy()

    assert c.keys() == h5_dict.keys()


def test_h5_dict_copy_nested_h5_dict_should_be_dict(h5_dict):
    c = h5_dict.copy()

    assert isinstance(c["c"], dict)


def test_h5_dict_copy_dataset_proxy_should_be_array(h5_dict):
    c = h5_dict.copy()

    assert type(c["b"]) == np.ndarray


class ComplexObject:
    def __init__(self, value: int):
        self.value = value

    def __repr__(self) -> str:
        return f"CO({self.value})"

    def __h5_write__(self, values: H5Dict) -> None:
        values.attributes["value"] = self.value

    @classmethod
    def __h5_read__(cls, values: H5Dict) -> ComplexObject:
        return ComplexObject(values.attributes["value"])


def test_h5_dict_can_store_complex_objects(h5_dict):
    h5_dict["g"] = {"a": ComplexObject(1), "b": ComplexObject(2)}

    assert isinstance(h5_dict["g"]["a"], ComplexObject)
