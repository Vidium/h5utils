# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import h5py
import numpy as np
from abc import ABC
from abc import abstractmethod
from numbers import Number
from h5py._hl.base import ValuesViewHDF5
from h5py._hl.base import ItemsViewHDF5

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import TypeVar
from typing import Generic

from ch5mpy._typing import SELECTOR
from ch5mpy.pickle.wrap import PickleableH5PyObject


# ====================================================
# code
_T = TypeVar('_T', bound=np.generic)


class _GroupManagerMixin(h5py.Group, ABC):
    """Mixin for File and Group objects which can access and create groups on H5 files."""

    @abstractmethod
    def _wrap(self, obj: Any) -> Any:
        """Wrap an object accessed in this group with our custom classes."""
        pass

    def __getitem__(self, name: str | bytes) -> Any:                                        # type: ignore[override]
        return self._wrap(h5py.Group.__getitem__(self, name))                               # type: ignore[index]

    def values(self) -> ValuesViewHDF5[Group | Dataset[Any]]:                               # type: ignore[override]
        return super().values()                                                             # type: ignore[return-value]

    def items(self) -> ItemsViewHDF5[str, Group | Dataset[Any]]:                            # type: ignore[override]
        return super().items()                                                              # type: ignore[return-value]

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


def _h5py_wrap_type(obj: Any) -> Any:
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


class Dataset(Generic[_T], PickleableH5PyObject, h5py.Dataset):
    """Mix in our pickling class"""

    def __getitem__(self,                                                                       # type: ignore[override]
                    arg: SELECTOR | tuple[SELECTOR, ...],
                    new_dtype: npt.DTypeLike | None = None) -> Number | str | npt.NDArray[_T]:
        return super().__getitem__(arg, new_dtype)

    def __setitem__(self, arg: SELECTOR | tuple[SELECTOR, ...], val: Any) -> None:
        super().__setitem__(arg, val)

    @property
    def dtype(self) -> np.dtype[_T]:
        return self.id.dtype                                                                # type: ignore[return-value]


class Group(PickleableH5PyObject, _GroupManagerMixin):
    """Overwrite group to allow pickling, and to create new groups and datasets
    of the right type (i.e. the ones defined in this module).
    """

    def _wrap(self, obj: Any) -> Any:
        obj = _h5py_wrap_type(obj)

        # If it is a group or dataset copy the current file info in
        if isinstance(obj, Group) or isinstance(obj, Dataset):
            obj.file_info = self.file_info

        return obj


class File(PickleableH5PyObject, _GroupManagerMixin, h5py.File):
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
        obj = _h5py_wrap_type(obj)

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
