from __future__ import annotations

from pathlib import Path
from typing import Any, Generator

import pytest

import ch5mpy as ch


class O_:
    def __init__(self, v: float):
        self._v = v

    def __repr__(self) -> str:
        return f"O({self._v})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, O_):
            return False

        return self._v == other._v


class C:
    def __init__(self, l_: list[int]):
        self.l = l_

    def __repr__(self) -> str:
        return f"C({self.l})"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, C):
            return False

        return self.l == other.l

    def __h5_write__(self, values: ch.H5Dict) -> None:
        values["l"] = self.l

    @classmethod
    def __h5_read__(cls, values: ch.H5Dict) -> C:
        return C(list(values["l"]))


@pytest.fixture
def h5_list() -> Generator[ch.H5List, None, None]:
    data = [1.0, 2, C([1, 2, 3]), "4.", O_(5.0)]

    with ch.File("h5_list.h5", ch.H5Mode.WRITE_TRUNCATE) as h5_file:
        ch.H5List.write(data, h5_file, "data")

    yield ch.H5List.read("h5_list.h5", "data", mode=ch.H5Mode.READ_WRITE)

    Path("h5_list.h5").unlink()


def test_list_repr(h5_list):
    assert repr(h5_list) == "H5List[1.0, 2, C([1, 2, 3]), '4.', O(5.0)]"


def test_list_should_read_custom_object(h5_list):
    assert isinstance(h5_list[4], O_)


def test_list_should_read_custom_object_with_method(h5_list):
    assert isinstance(h5_list[2], C)


def test_list_should_convert_to_regular_list(h5_list):
    lst = h5_list.copy()
    assert lst == [1.0, 2, C([1, 2, 3]), "4.", O_(5.0)]


def test_list_get_negative_index(h5_list):
    assert h5_list[-2] == "4."


def test_can_append_value_to_list(h5_list):
    h5_list.append(-1)
    assert h5_list[5] == -1
