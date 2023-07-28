from __future__ import annotations

from pathlib import Path
from typing import Any, Generic, Iterator, TypeVar, cast

from ch5mpy.dict import H5Dict
from ch5mpy.names import H5Mode
from ch5mpy.objects import File, Group, H5Object
from ch5mpy.read import read_object
from ch5mpy.write import write_object

_T = TypeVar("_T")


def _get_group(file: File | Group, name: str) -> tuple[File | Group, str]:
    lst = name.rsplit("/", 1)

    if len(lst) == 1:
        return file, lst[0]

    return file.require_group(lst[0]), lst[1]


class H5List(H5Object, Generic[_T]):
    """Class for managing lists backed on h5 files."""

    # region magic methods
    def __init__(self, file: File | Group):
        super().__init__(file)
        self._iter_index = 0

    def __repr__(self) -> str:
        if self.is_closed:
            return "Closed H5List[...]"

        if len(self) > 6:
            return f"H5List[{self[0]}, {self[1]}, {self[2]}, ..., " f"{self[-3]}, {self[-2]}, {self[-1]}]"

        return f"H5List[{', '.join([repr(e) for e in self])}]"

    def __len__(self) -> int:
        return len(self._file.keys())

    def __getitem__(self, item: int) -> _T:
        if item < 0:
            item = len(self) + item

        return cast(_T, read_object(self._file[str(item)]))

    def __iter__(self) -> Iterator[_T]:
        self._iter_index = 0
        return self

    def __next__(self) -> _T:
        if self._iter_index >= len(self):
            raise StopIteration

        obj = cast(_T, read_object(self._file[str(self._iter_index)]))
        self._iter_index += 1

        return obj

    # endregion

    # region class methods
    @classmethod
    def read(cls, path: str | Path | File | Group, name: str = "", mode: H5Mode = H5Mode.READ) -> H5List[Any]:
        if isinstance(path, (str, Path)):
            path = File(path, mode=mode)

        src = path[name]

        if not isinstance(src, Group):
            raise ValueError(f"Cannot read H5List from '{type(src)}' object, " f"should be a 'Group'.")

        src_type = src.attrs.get("__h5_type__", "")

        if src_type != "list":
            raise ValueError(f"Cannot read H5List from Group marked as type " f"'{src_type}', should be 'list'.")

        return H5List(src)

    @classmethod
    def write(cls, lst: list[Any], path: str | Path | File | Group, name: str = "") -> None:
        """
        Write a list to a .h5 file as a H5List.

        Args:
            lst: a list to write.
            path: Either a path to the .h5 file or an open ch5mpy.File or
                ch5mpy.Group object.
            name: The group member name inside the .h5 file to use.
        """
        if isinstance(path, (str, Path)):
            path = File(path, mode=H5Mode.READ_WRITE)

        if path.file.mode == H5Mode.READ:
            raise ValueError("Cannot write to h5 file open in 'r' mode.")

        dest, group_name = _get_group(path, name)

        if group_name in dest.keys():
            raise ValueError(f"An object with name '{group_name}' already exists.")

        dest = dest.create_group(group_name)
        dest.attrs["__h5_type__"] = "list"

        for index, obj in enumerate(lst):
            write_object(dest, str(index), obj, chunks=True)

    # endregion

    # region methods
    def copy(self) -> Any:
        return [e for e in self]

    def to_dict(self) -> H5Dict[_T]:
        return H5Dict(self._file)

    def append(self, value: None) -> None:
        write_object(self._file, str(len(self)), value, chunks=True)

    # endregion
