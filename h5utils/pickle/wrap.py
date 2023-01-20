# coding: utf-8

"""
Modify h5 File, Group and Dataset objects to allow pickling.

Modified from https://github.com/DaanVanVugt/h5pickle/blob/master/h5pickle
"""

# ====================================================
# imports
from __future__ import annotations

import h5py
from abc import ABC
from abc import abstractmethod

from typing import Any
from typing import cast
from typing import TypeVar


# ====================================================
# code
X = TypeVar("X")


class PickleableH5PyObject(h5py.HLObject):
    """Save state required to pickle and unpickle h5py objects and groups.
    This class should not be used directly, but is here as a base for inheritance
    for Group and Dataset"""

    def __getstate__(self) -> dict[str, Any] | None:
        """Save the current name and a reference to the root file object."""
        return {"name": self.name, "file": self.file_info}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """File is reopened by pickle. Create an object and steal its identity."""
        # we re-create the object by calling __init__, this is technically unsafe, but I found no alternative for now
        self.__init__(state["file"][state["name"]].id)  # type: ignore[misc]
        self.file_info = state["file"]

    def __getnewargs__(self) -> tuple[()]:
        """Override the h5py getnewargs to skip its error message"""
        return ()


class GroupManagerMixin(h5py.Group, ABC):
    """Mixin for File and Group objects which can access and create groups on H5 files."""

    @abstractmethod
    def _wrap(self, obj: Any) -> Any:
        """Wrap an object accessed in this group with our custom classes."""
        pass

    def __getitem__(self, name: str | slice | tuple[()]) -> Any:
        return self._wrap(h5py.Group.__getitem__(self, name))

    def create_group(self, name: str, track_order: bool | None = None) -> Group:
        """
        Create and return a new subgroup.

        Args:
            name: may be absolute or relative. Fails if the target name already exists.
            track_order: Track dataset/group/attribute creation order under this group if True. If None use global
            default h5.get_config().track_order.
        """
        group = super().create_group(name, track_order=track_order)

        return cast(Group, self._wrap(group))


class Dataset(PickleableH5PyObject, h5py.Dataset):
    """Mix in our pickling class"""

    pass


class Group(PickleableH5PyObject, GroupManagerMixin):
    """Overwrite group to allow pickling, and to create new groups and datasets
    of the right type (i.e. the ones defined in this module).
    """

    def _wrap(self, obj: Any) -> Any:
        obj = h5py_wrap_type(obj)

        # If it is a group or dataset copy the current file info in
        if isinstance(obj, Group) or isinstance(obj, Dataset):
            obj.file_info = self.file_info

        return obj


class File(PickleableH5PyObject, GroupManagerMixin, h5py.File):
    """A subclass of h5py.File that implements pickling.
    Pickling is done not with __{get,set}state__ but with __getnewargs_ex__
    which produces the arguments to supply to the __new__ method.
    """

    # noinspection PyMissingConstructor
    def __init__(self, *args: Any, **kwargs: Any):
        # Store args and kwargs for pickling
        self.init_args = args
        self.init_kwargs = kwargs

    def __new__(cls, *args: Any, **kwargs: Any) -> File:
        """Create a new File object with the h5 open function."""
        self = super().__new__(cls)
        h5py.File.__init__(self, *args, **kwargs)

        return self

    def _wrap(self, obj: Any) -> Any:
        obj = h5py_wrap_type(obj)

        # If it is a group or dataset copy the current file info in
        if isinstance(obj, Group) or isinstance(obj, Dataset):
            obj.file_info = self

        return obj

    def __getstate__(self) -> None:
        pass

    def __getnewargs_ex__(self) -> tuple[tuple[Any, ...], dict[str, Any]]:
        kwargs = self.init_kwargs.copy()

        if len(self.init_args) > 1 and self.init_args[1] == "w":
            return (self.init_args[0], "r+", *self.init_args[2:]), kwargs

        return self.init_args, kwargs


def h5py_wrap_type(obj: Any) -> Any:
    """Produce our objects instead of h5py default objects"""
    if isinstance(obj, h5py.Dataset):
        return Dataset(obj.id)
    elif isinstance(obj, h5py.File):
        return File(obj.id)
    elif isinstance(obj, h5py.Group):
        return Group(obj.id)
    elif isinstance(obj, h5py.Datatype):
        return obj  # Not supported for pickling yet. Haven't really thought about it
    else:
        return obj  # Just return, since we want to wrap h5py.Group.get too
