import pytest
from tempfile import NamedTemporaryFile

from h5utils import File


@pytest.fixture
def group():
    tmp = NamedTemporaryFile()

    with File(tmp.name, "r+") as h5_file:
        yield h5_file.create_group("test")
