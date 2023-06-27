from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Generator

import numpy as np
import pytest

from ch5mpy import File, H5Array, H5Mode, write_object


@pytest.fixture
def group():
    tmp = NamedTemporaryFile()

    with File(tmp.name, "r+") as h5_file:
        yield h5_file.create_group("test")


@pytest.fixture
def small_array() -> Generator[H5Array, None, None]:
    data = [1.0, 2.0, 3.0, 4.0, 5.0]

    with File("h5_s_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_s_array", H5Mode.READ_WRITE)["data"])

    Path("h5_s_array").unlink()


@pytest.fixture
def empty_array() -> Generator[H5Array, None, None]:
    with File("h5_e_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", np.empty((0, 1)))

    yield H5Array(File("h5_e_array", H5Mode.READ_WRITE)["data"])

    Path("h5_e_array").unlink()


@pytest.fixture
def array() -> Generator[H5Array, None, None]:
    data = np.arange(100.0).reshape((10, 10))

    with File("h5_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_array", H5Mode.READ_WRITE)["data"])

    Path("h5_array").unlink()


@pytest.fixture
def chunked_array() -> Generator[H5Array, None, None]:
    data = np.arange(100.0).reshape((10, 10))

    with File("h5_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data, chunks=(3, 3))

    yield H5Array(File("h5_array", H5Mode.READ_WRITE)["data"])

    Path("h5_array").unlink()


@pytest.fixture
def small_large_array() -> Generator[H5Array, None, None]:
    data = np.arange(3 * 4 * 5).reshape((3, 4, 5))

    with File("h5_sl_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_sl_array", H5Mode.READ_WRITE)["data"])

    Path("h5_sl_array").unlink()


@pytest.fixture
def large_array() -> Generator[H5Array, None, None]:
    data = np.arange(20_000 * 10_000).reshape((20_000, 10_000))

    with File("h5_large_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_large_array", H5Mode.READ_WRITE)["data"])

    Path("h5_large_array").unlink()


@pytest.fixture
def str_array() -> Generator[H5Array, None, None]:
    data = ["a", "bc", "d", "efg", "h"]

    with File("h5_str_array", H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(h5_file, "data", data)

    yield H5Array(File("h5_str_array", H5Mode.READ_WRITE)["data"])

    Path("h5_str_array").unlink()
