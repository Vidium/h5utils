# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
from numbers import Number
from collections.abc import KeysView
from collections.abc import Iterable
from collections.abc import Collection
from collections.abc import MutableMapping
from h5py._hl.base import ItemsViewHDF5

from typing import Any
from typing import cast
from typing import Union
from typing import Iterator

import ch5mpy as hu
from ch5mpy.objects import File
from ch5mpy.objects import Group
from ch5mpy.objects import Dataset
from ch5mpy.write import write_object

# ====================================================
# code
H5DICT_CONTENT = Union["H5Dict", "hu.H5Array[Any]", Number, str]


def _get_in_memory(value: Any) -> Any:
    if isinstance(value, H5Dict):
        return value.copy()

    elif isinstance(value, Dataset):
        return value[()]

    return value


def _get_repr(items: ItemsViewHDF5[str, Group | Dataset[Any]]) -> str:
    return ", ".join([str(k) + ": " + (repr(_parse_value(v)) if not isinstance(v, Group) else "{...}")
                      for k, v in items])


class H5Dict(MutableMapping[str, 'H5DICT_CONTENT']):
    """Class for managing dictionaries backed on h5 files."""

    # region magic methods
    def __init__(self, file: File | Group):
        assert isinstance(file, (File, Group))

        self._file = file

    def __dir__(self) -> Iterable[str]:
        return dir(H5Dict) + list(self.keys())

    def __repr__(self) -> str:
        if self.is_closed:
            return "Closed H5Dict{}"

        return f"H5Dict{'{' + _get_repr(self._file.items()) + '}'}"

    def __setitem__(self, key: str, value: Any) -> None:
        write_object(self._file, key, value)

    def __delitem__(self, key: str) -> None:
        del self._file[key]

    def __getitem__(self, key: str) -> H5DICT_CONTENT:
        return _parse_value(self._file[key])

    def __getattr__(self, item: str) -> H5DICT_CONTENT:
        return self.__getitem__(item)

    def __len__(self) -> int:
        return len(self._file.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    # endregion

    # region predicates
    @property
    def is_closed(self) -> bool:
        return not self._file.id.valid

    # endregion

    # region methods
    def keys(self) -> KeysView[str]:
        return self._file.keys()

    def values(self) -> H5DictValuesView:                           # type: ignore[override]
        # from typing import reveal_type
        return H5DictValuesView(self._file.values())

    def items(self) -> H5DictItemsView:                             # type: ignore[override]
        return H5DictItemsView(self._file.keys(), self._file.values())

    def close(self) -> None:
        self._file.file.close()

    def copy(self) -> dict[str, Any]:
        """
        Build an in-memory copy of this H5Dict object.
        """
        return {k: _get_in_memory(v) for k, v in self.items()}

    # endregion


def _parse_value(obj: Group | Dataset[Any]) -> H5DICT_CONTENT:
    """Parse a h5 object into a higher abstraction-level object."""
    if isinstance(obj, Group):
        # return Group as H5Dict
        return H5Dict(obj)

    elif isinstance(obj, Dataset):
        if obj.shape == ():
            # single value in dataset
            if np.issubdtype(obj.dtype, np.number) or np.issubdtype(
                obj.dtype, np.bool_
            ):
                # return numeric value
                return cast(Number, obj[()])

            else:
                # return string value
                return cast(str, obj.asstr()[()])

        # return dataset as H5Array
        return hu.H5Array(obj)

    else:
        raise ValueError(
            f"Got unexpected object of type '{type(obj)}' for key '{obj}'."
        )


class H5DictValuesView(Iterable[H5DICT_CONTENT]):
    """Class for iterating over values in an H5Dict."""

    # region magic methods
    def __init__(self, values: Collection[Group | Dataset[Any]]):
        self._values = values

    def __repr__(self) -> str:
        return f"{type(self).__name__}([{len(self._values)} values])"

    def __iter__(self) -> Iterator[H5DICT_CONTENT]:
        return map(_parse_value, self._values)

    # endregion


class H5DictItemsView(Iterable[tuple[str, H5DICT_CONTENT]]):
    """Class for iterating over items in an H5Dict."""

    # region magic methods
    def __init__(self, keys: Collection[str], values: Collection[Group | Dataset[Any]]):
        self._keys = keys
        self._values = values

    def __repr__(self) -> str:
        return f"{type(self).__name__}([{len(self._keys)} items])"

    def __iter__(self) -> Iterator[tuple[str, H5DICT_CONTENT]]:
        return zip(self._keys, map(_parse_value, self._values))

    # endregion
