# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from collections.abc import Collection, Iterable, KeysView, MutableMapping
from functools import partial
from pathlib import Path
from typing import Any, Iterator, TypeVar, cast

import numpy as np
from h5py._hl.base import ItemsViewHDF5

from ch5mpy.h5array import H5Array
from ch5mpy.names import H5Mode
from ch5mpy.objects.dataset import AsStrWrapper, Dataset
from ch5mpy.objects.group import File, Group
from ch5mpy.objects.object import H5Object
from ch5mpy.read import read_object
from ch5mpy.write import write_object

# ====================================================
# code
_T = TypeVar("_T")

_safe_read = partial(read_object, error="ignore")
_NO_OBJECT = object()


def _get_in_memory(value: Any) -> Any:
    if isinstance(value, H5Dict):
        return value.copy()

    elif isinstance(value, (Dataset, AsStrWrapper)):
        return value[()]

    elif isinstance(value, (H5Object, H5Array)):
        return value.copy()

    return value


def _is_group(obj: Group | Dataset[Any]) -> bool:
    h5_type = obj.attrs.get("__h5_type__", "<UNKNOWN>")
    return isinstance(obj, Group) and h5_type != "object"


def _get_repr(items: ItemsViewHDF5[str, Group | Dataset[Any]]) -> str:
    if not len(items):
        return "{}"

    return (
        "{\n\t"
        + ",\n\t".join(
            [str(k) + ": " + ("{...}" if _is_group(v) else repr(_safe_read(v)).replace("\n", "\n\t")) for k, v in items]
        )
        + "\n}"
    )


def _get_note(annotation: str | None) -> str:
    return "" if annotation is None else f"[{annotation}]"


class H5DictValuesView(Iterable[_T]):
    """Class for iterating over values in an H5Dict."""

    # region magic methods
    def __init__(self, values: Collection[Group | Dataset[Any]]):
        self._values = values

    def __repr__(self) -> str:
        return f"{type(self).__name__}([{len(self._values)} values])"

    def __iter__(self) -> Iterator[_T]:
        return map(_safe_read, self._values)

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
        return zip(self._keys, map(_safe_read, self._values))

    # endregion


def _diff(a: Any, b: Any) -> bool:
    return bool(np.array(a != b).any())


class H5Dict(H5Object, MutableMapping[str, _T]):
    """Class for managing dictionaries backed on h5 files."""

    # region magic methods
    def __init__(self, file: File | Group, annotation: str | None = None):
        super().__init__(file)
        self.annotation = annotation

    def __dir__(self) -> Iterable[str]:
        return dir(H5Dict) + list(self.keys())

    def __repr__(self) -> str:
        if self.is_closed:
            return "Closed H5Dict{}"

        return f"H5Dict{_get_note(self.annotation)}{_get_repr(self._file.items())}"

    def __getitem__(self, key: str) -> _T:
        return cast(_T, _safe_read(self._file[key]))

    def __setitem__(self, key: str, value: Any) -> None:
        if callable(value):
            value(name=key, loc=self._file)

        elif _diff(self.get(key, _NO_OBJECT), value):
            write_object(self, key, value, overwrite=True)

    def __delitem__(self, key: str) -> None:
        del self._file[key]

    def __getattr__(self, item: str) -> _T:
        return self.__getitem__(item)

    def __len__(self) -> int:
        return len(self._file.keys())

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    # endregion

    # region class methods
    @classmethod
    def read(cls, path: str | Path | File | Group, name: str | None = None, mode: H5Mode = H5Mode.READ) -> H5Dict[Any]:
        file = File(path, mode=mode) if isinstance(path, (str, Path)) else path

        if name is not None:
            file = file[name]

        return H5Dict(file)

    # endregion

    # region methods
    def keys(self) -> KeysView[str]:
        return self._file.keys()

    def values(self) -> H5DictValuesView[_T]:  # type: ignore[override]
        # from typing import reveal_type
        return H5DictValuesView(self._file.values())

    def items(self) -> H5DictItemsView[_T]:  # type: ignore[override]
        return H5DictItemsView(self._file.keys(), self._file.values())

    def get(self, key: str, default: Any = None) -> Any:
        res = self._file.get(key, default)

        if res is default:
            return default
        return _safe_read(res)

    def rename(self, name: str, new_name: str) -> None:
        self._file.move(name, new_name)

    def copy(self) -> dict[str, Any]:
        """
        Build an in-memory copy of this H5Dict object.
        """
        return {k: _get_in_memory(v) for k, v in self.items()}

    # endregion