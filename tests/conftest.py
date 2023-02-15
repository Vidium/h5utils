import pytest
import numpy as np
from pathlib import Path
from tempfile import NamedTemporaryFile

from ch5mpy import H5Array
from ch5mpy import File
from ch5mpy import H5Mode
from ch5mpy import write_object


@pytest.fixture
def group():
    tmp = NamedTemporaryFile()

    with File(tmp.name, "r+") as h5_file:
        yield h5_file.create_group("test")


@pytest.fixture
def small_array() -> H5Array:
    data = [1., 2., 3., 4., 5.]

    with File("h5_s_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_s_array", H5Mode.READ_WRITE)["data"])

    Path("h5_s_array").unlink()


@pytest.fixture
def array() -> H5Array:
    data = np.arange(100.).reshape((10, 10))

    with File("h5_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_array", H5Mode.READ_WRITE)["data"])

    Path("h5_array").unlink()


@pytest.fixture
def small_large_array() -> H5Array:
    data = np.arange(3 * 4 * 5).reshape((3, 4, 5))

    with File("h5_sl_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_sl_array", H5Mode.READ_WRITE)["data"])

    Path("h5_sl_array").unlink()


@pytest.fixture
def large_array() -> H5Array:
    data = np.arange(20_000 * 10_000).reshape((20_000, 10_000))

    with File("h5_large_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_large_array", H5Mode.READ_WRITE)["data"])

    Path("h5_large_array").unlink()
