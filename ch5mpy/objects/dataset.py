# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import h5py
import numpy as np
from numbers import Number
from h5py.h5t import check_string_dtype

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import TypeVar
from typing import Generic
from typing import Literal

from ch5mpy._typing import SELECTOR
from ch5mpy.pickle.wrap import PickleableH5PyObject

# ====================================================
# code
_T = TypeVar('_T', bound=np.generic)
ENCODING = Literal['ascii', 'utf-8']
ERROR_METHOD = Literal['backslashreplace', 'ignore', 'namereplace', 'strict', 'replace', 'xmlcharrefreplace']


class AsStrWrapper:
    """Wrapper to decode strings on reading the dataset"""

    # region magic methods
    def __init__(self, dset: Dataset[np.bytes_]):
        self._dset = dset

    def __getitem__(self, args: SELECTOR | tuple[SELECTOR, ...]) -> npt.NDArray[np.str_]:
        return np.array(self._dset[args], dtype=str)

    def __getattr__(self, attr: str) -> Any:
        # If method/attribute is not defined here, pass the call to the wrapped dataset.
        return getattr(self._dset, attr)

    def __len__(self) -> int:
        return len(self._dset)

    # endregion

    # region numpy interface
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        return self[()]

    # endregion

    # region attributes
    @property
    def dtype(self) -> np.dtype[np.str_]:
        max_str_len = len(max(self._dset, key=len))                                        # type: ignore[call-overload]
        return np.dtype('<U' + str(max_str_len))                    # FIXME : is there a better way to find out the
                                                                    #  largest string ?

    @property
    def size(self) -> int:
        return self._dset.size

    # endregion


class Dataset(Generic[_T], PickleableH5PyObject, h5py.Dataset):
    """Mix in our pickling class"""

    # region magic methods
    def __getitem__(self,                                                                       # type: ignore[override]
                    arg: SELECTOR | tuple[SELECTOR, ...],
                    new_dtype: npt.DTypeLike | None = None) -> Number | str | npt.NDArray[_T]:
        return super().__getitem__(arg, new_dtype)

    def __setitem__(self, arg: SELECTOR | tuple[SELECTOR, ...], val: Any) -> None:
        super().__setitem__(arg, val)

    # endregion

    # region attributes
    @property
    def dtype(self) -> np.dtype[_T]:
        return self.id.dtype                                                                # type: ignore[return-value]

    # endregion

    # region methods
    def asstr(self,                                                                             # type: ignore[override]
              encoding: ENCODING | None = None,
              errors: ERROR_METHOD = 'strict') -> AsStrWrapper:
        """
        Get a wrapper to read string data as Python strings:

        The parameters have the same meaning as in ``bytes.decode()``.
        If ``encoding`` is unspecified, it will use the encoding in the HDF5
        datatype (either ascii or utf-8).
        """
        string_info = check_string_dtype(self.dtype)
        if string_info is None:
            raise TypeError("dset.asstr() can only be used on datasets with an HDF5 string datatype")

        return AsStrWrapper(cast(Dataset[np.bytes_], self))

    # endregion
