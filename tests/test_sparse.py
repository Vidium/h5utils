import pickle
from tempfile import NamedTemporaryFile

import numpy as np
import pytest
import scipy.sparse as sp

from ch5mpy import File, Group, H5Mode, write_object
from ch5mpy.array.array import H5Array
from ch5mpy.io import read_object
from ch5mpy.sparse.csr import H5_csr_array


@pytest.fixture
def h5_csr_data():
    data = {
        "data": [1, 2, 3, 4],
        "indices": [1, 2, 0, 3],
        "indptr": [0, 1, 2, 4],
    }

    tmp = NamedTemporaryFile()

    with File(tmp.name, H5Mode.WRITE_TRUNCATE) as h5_file:
        write_object(data, h5_file, "csr")
        h5_file["csr"].attrs["__h5_type__"] = "object"
        h5_file["csr"].attrs["__h5_class__"] = np.void(pickle.dumps(H5_csr_array, protocol=pickle.HIGHEST_PROTOCOL))
        h5_file["csr"].attrs["_shape"] = (3, 4)

        yield h5_file["csr"]


@pytest.fixture
def h5_csr(h5_csr_data: Group):
    return read_object(h5_csr_data)


def test_csr_creation_from_file(h5_csr_data: Group):
    csr = read_object(h5_csr_data)
    assert isinstance(csr, H5_csr_array)
    assert isinstance(csr.data, H5Array)
    assert (csr != sp.csr_matrix(np.array([[0, 1, 0, 0], [0, 0, 2, 0], [3, 0, 0, 4]]))).nnz == 0


def test_csr_can_multiply(h5_csr: H5_csr_array):
    assert np.array_equal((h5_csr * 2).todense(), np.array([[0, 2, 0, 0], [0, 0, 4, 0], [6, 0, 0, 8]]))


def test_csr_can_set_value(h5_csr: H5_csr_array):
    h5_csr[0, 3] = 5
    assert isinstance(h5_csr.data, H5Array)
    assert np.array_equal(h5_csr.data, [1, 5, 2, 3, 4])
    assert np.array_equal(h5_csr.todense(), np.array([[0, 1, 0, 5], [0, 0, 2, 0], [3, 0, 0, 4]]))


def test_csr_can_set_many_values(h5_csr: H5_csr_array):
    h5_csr._insert_many(np.array([0, 4]), np.array([3, 0]), np.array([5, 6]))
    assert np.array_equal(
        h5_csr.todense(), np.array([[0, 1, 0, 5], [0, 0, 2, 0], [3, 0, 0, 4], [0, 0, 0, 0], [6, 0, 0, 0]])
    )
