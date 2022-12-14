# coding: utf-8
# Created on 13/12/2022 14:52
# Author : matteo

# ====================================================
# imports
import pickle
import pytest
import numpy as np
from h5py import File
from tempfile import NamedTemporaryFile

from h5utils.write import write_attribute
from enum import Enum


# ====================================================
# code
class State(Enum):
    RNA = 'RNA'


@pytest.mark.parametrize(
    'obj, expected',
    [(1, 1),
     ('abc', 'abc'),
     (None, 'None')]
)
def test_should_write_simple_attribute(obj, expected):
    tmp = NamedTemporaryFile()

    with File(tmp.name, 'r+') as h5_file:
        group = h5_file.create_group('test')

        write_attribute(group, 'something', obj)

        assert 'something' in group.attrs.keys()
        assert group.attrs['something'] == expected


def test_should_write_list_attribute():
    tmp = NamedTemporaryFile()

    with File(tmp.name, 'r+') as h5_file:
        group = h5_file.create_group('test')

        write_attribute(group, 'something', [1, 2, 3])

        assert 'something' in group.attrs.keys()
        assert np.all(group.attrs['something'] == [1, 2, 3])


def test_should_write_complex_objects_as_strings():
    tmp = NamedTemporaryFile()

    with File(tmp.name, 'r+') as h5_file:
        group = h5_file.create_group('test')

        write_attribute(group, 'something', State.RNA)

        assert 'something' in group.attrs.keys()
        assert group.attrs['something'].tostring() == pickle.dumps(State.RNA)
