# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from collections.abc import Collection, Iterable, KeysView, MutableMapping
from pathlib import Path
from typing import Any, Iterator, TypeVar, cast

from h5py._hl.base import ItemsViewHDF5

from ch5mpy.names import H5Mode
from ch5mpy.objects.dataset import AsStrWrapper, Dataset
from ch5mpy.objects.group import File, Group
from ch5mpy.objects.object import H5Object
from ch5mpy.read import read_object
from ch5mpy.write import write_object

# ====================================================
# code
_T = TypeVar("_T")


def _get_in_memory(value: Any) -> Any:
    if isinstance(value, H5Dict):
        return value.copy()

    elif isinstance(value, (Dataset, AsStrWrapper)):
        return value[()]

    return value


def _parse_value(obj: Group | Dataset[Any]) -> Any:
    """Parse a h5 object into a higher abstraction-level object."""
    if isinstance(obj, Group):
        # return Group as H5Dict
        return H5Dict(obj)

    elif isinstance(obj, Dataset):
        return read_object(obj)

    else:
        raise ValueError(f"Got unexpected object of type '{type(obj)}' for key '{obj}'.")


def _get_repr(items: ItemsViewHDF5[str, Group | Dataset[Any]]) -> str:
    if not len(items):
        return "{}"

    return (
        "{\n\t"
        + ",\n\t".join([str(k) + ": " + ("{...}" if isinstance(v, Group) else repr(_parse_value(v))) for k, v in items])
        + "\n}"
    )


class H5DictValuesView(Iterable[_T]):
    """Class for iterating over values in an H5Dict."""

    # region magic methods
    def __init__(self, values: Collection[Group | Dataset[Any]]):
        self._values = values

    def __repr__(self) -> str:
        return f"{type(self).__name__}([{len(self._values)} values])"

    def __iter__(self) -> Iterator[_T]:
        return map(_parse_value, self._values)

    # endregion


class H5DictItemsView(Iterable[tuple[str, _T]]):
    """Class for iterating over items in an H5Dict."""

    # region magic methods
    def __init__(self, keys: Collection[str], values: Collection[Group | Dataset[Any]]):
        self._keys = keys
        self._values = values

    def __repr__(self) -> str:
        return f"{type(self).__name__}([{len(self._keys)} items])"

    def __iter__(self) -> Iterator[tuple[str, _T]]:
        return zip(self._keys, map(_parse_value, self._values))

    # endregion


class H5Dict(H5Object, MutableMapping[str, _T]):
    """Class for managing dictionaries backed on h5 files."""

    # region magic methods
    def __dir__(self) -> Iterable[str]:
        return dir(H5Dict) + list(self.keys())

    def __repr__(self) -> str:
        if self.is_closed:
            return "Closed H5Dict{}"

        return f"H5Dict{_get_repr(self._file.items())}"

    def __setitem__(self, key: str, value: Any) -> None:
        if callable(value):
            value(name=key, loc=self._file)

        else:
            write_object(self._file, key, value, overwrite=True)

    def __delitem__(self, key: str) -> None:
        del self._file[key]

    def __getitem__(self, key: str) -> _T:
        return cast(_T, _parse_value(self._file[key]))

    def __getattr__(self, item: str) -> _T:
        return self.__getitem__(item)

    def __len__(self) -> int:
        return len(self._file.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    # endregion

    # region class methods
    @classmethod
    def read(cls, path: str | Path | File | Group, name: str = "", mode: H5Mode = H5Mode.READ) -> H5Dict[Any]:
        if isinstance(path, (str, Path)):
            path = File(path, mode=mode)

        return H5Dict(path[name])

    # endregion

    # region methods
    def keys(self) -> KeysView[str]:
        return self._file.keys()

    def values(self) -> H5DictValuesView[_T]:  # type: ignore[override]
        # from typing import reveal_type
        return H5DictValuesView(self._file.values())

    def items(self) -> H5DictItemsView[_T]:  # type: ignore[override]
        return H5DictItemsView(self._file.keys(), self._file.values())

    def copy(self) -> dict[str, Any]:
        """
        Build an in-memory copy of this H5Dict object.
        """
        return {k: _get_in_memory(v) for k, v in self.items()}

    # endregion
