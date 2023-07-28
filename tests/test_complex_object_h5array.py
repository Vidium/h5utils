import pytest
from pathlib import Path

from ch5mpy import File
from ch5mpy import H5Mode
from ch5mpy import H5Array
from ch5mpy import write_object


class ComplexObject:
    def __init__(self, value: int):
        self.value = value

    def __repr__(self) -> str:
        return f"CO({self.value})"


@pytest.fixture
def co_array() -> H5Array:
    data = [1, 2, 3, 4, 5, 6, 7]

    with File("h5_str_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_str_array", H5Mode.READ_WRITE)["data"]).maptype(ComplexObject)

    Path("h5_str_array").unlink()


def test_co_array_dtype(co_array):
    assert co_array.dtype == object


def test_co_array_equals(co_array):
    first_element = co_array[0]
    assert isinstance(first_element, ComplexObject) and first_element.value == 1


def test_co_array_should_repr_str(co_array):
    assert repr(co_array) == "H5Array([CO(1), CO(2), CO(3), ..., CO(5), CO(6), CO(7)], shape=(7,), dtype=object)"
