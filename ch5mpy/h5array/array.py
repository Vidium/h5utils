# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from numbers import Number
from typing import TYPE_CHECKING, Any, Generic, Iterator, TypeVar

import numpy as np
import numpy.lib.mixins
import numpy.typing as npt

from ch5mpy._typing import NP_FUNC, SELECTOR
from ch5mpy.h5array import repr
from ch5mpy.h5array.chunks.iter import ChunkIterator, PairedChunkIterator
from ch5mpy.h5array.functions import HANDLED_FUNCTIONS
from ch5mpy.h5array.io import read_one_from_dataset, write_to_dataset
from ch5mpy.indexing import Selection, map_slice
from ch5mpy.objects.dataset import Dataset, DatasetWrapper
from ch5mpy.write import write_dataset

if TYPE_CHECKING:
    from ch5mpy.h5array.view import H5ArrayView


# ====================================================
# code
_T = TypeVar("_T", bound=np.generic)
SIZES = {"K": 1024, "M": 1024 * 1024, "G": 1024 * 1024 * 1024}


def get_size(s: int | str) -> int:
    value: int | None = None

    if isinstance(s, int):
        value = s

    elif s[-1] in SIZES and s[:-1].lstrip("-").isdigit():
        value = int(s[:-1]) * SIZES[s[-1]]

    elif s.isdigit():
        value = int(s)

    if value is None:
        raise ValueError(f"Unrecognized size '{s}'")

    if value <= 0:
        raise ValueError(f"Got invalid size ({value} <= 0).")

    return value


def as_array(values: Any, dtype: np.dtype[Any]) -> npt.NDArray[Any]:
    # FIXME : work on H5Arrays directly instead of converting to np.array
    if np.issubdtype(dtype, str):
        return np.array(values, dtype=bytes)

    try:
        return np.array(values, dtype=dtype)

    except ValueError:
        raise ValueError(f"Couldn't set value of type {type(values)} in H5Array of type {dtype}.")


def _dtype_repr(dset: Dataset[Any] | DatasetWrapper[Any]) -> str:
    if np.issubdtype(dset.dtype, np.str_):
        return f"'{dset.dtype}'"

    return str(dset.dtype)


class H5Array(Generic[_T], numpy.lib.mixins.NDArrayOperatorsMixin):
    """Wrapper around Dataset objects to interface with numpy's API."""

    MAX_MEM_USAGE: int | str = "250M"

    # region magic methods
    def __init__(self, dset: Dataset[_T] | DatasetWrapper[_T]):
        if not isinstance(dset, (Dataset, DatasetWrapper)):
            raise TypeError(f"Object of type '{type(dset)}' is not supported by H5Array.")

        if isinstance(dset, Dataset) and np.issubdtype(np.dtype(str(dset.attrs.get("dtype", "O"))), str):
            self._dset: Dataset[_T] | DatasetWrapper[_T] = dset.asstr()  # type: ignore[assignment]

        else:
            self._dset = dset

    def __repr__(self) -> str:
        return (
            f"H5Array({repr.print_dataset(self, end='', padding=8, padding_skip_first=True)}, "
            f"shape={self.shape}, dtype={_dtype_repr(self._dset)})"
        )

    def __str__(self) -> str:
        return repr.print_dataset(self, sep="")

    def __getitem__(self, index: SELECTOR | tuple[SELECTOR, ...]) -> _T | H5Array[_T] | H5ArrayView[_T]:
        from ch5mpy.h5array.view import H5ArrayView

        selection = Selection.from_selector(index, self.shape)

        if selection.is_empty:
            return H5Array(dset=self._dset)

        elif selection.compute_shape(self._dset.shape) == ():
            return read_one_from_dataset(self._dset, selection, self.dtype)

        else:
            return H5ArrayView(dset=self._dset, sel=selection)

    def __setitem__(self, index: SELECTOR | tuple[SELECTOR, ...], value: Any) -> None:
        selection = Selection.from_selector(index, self.shape)
        write_to_dataset(self._dset, as_array(value, self.dtype), selection)

    def __len__(self) -> int:
        return len(self._dset)

    def __iter__(
        self,
    ) -> Iterator[_T | npt.NDArray[_T] | H5Array[_T] | H5ArrayView[_T]]:
        for i in range(self.shape[0]):
            yield self[i]

    def __contains__(self, item: Any) -> bool:
        for _, chunk in self.iter_chunks():
            if item in chunk:
                return True

        return False

    def _inplace(self, func: NP_FUNC, value: Any) -> H5Array[_T]:
        if np.issubdtype(self.dtype, str):
            raise TypeError("Cannot perform inplace operation on str H5Array.")

        # special case : 0D array
        if self.shape == ():
            self._dset[:] = func(self._dset[:], value)
            return self

        # general case : 1D+ array
        for index, chunk in self.iter_chunks():
            func(chunk, value, out=chunk)

            # write back result into array
            self.dset.write_direct(
                chunk,
                source_sel=map_slice(index, shift_to_zero=True),
                dest_sel=map_slice(index),
            )

        return self

    def __add__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return np.array(self) + other  # type: ignore[no-any-return]

    def __iadd__(self, other: Any) -> H5Array[_T]:
        return self._inplace(np.add, other)

    def __sub__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return np.array(self) - other  # type: ignore[no-any-return]

    def __isub__(self, other: Any) -> H5Array[_T]:
        return self._inplace(np.subtract, other)

    def __mul__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return np.array(self) * other  # type: ignore[no-any-return]

    def __imul__(self, other: Any) -> H5Array[_T]:
        return self._inplace(np.multiply, other)

    def __truediv__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return np.array(self) / other  # type: ignore[no-any-return]

    def __itruediv__(self, other: Any) -> H5Array[_T]:
        return self._inplace(np.divide, other)

    def __mod__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return np.array(self) % other  # type: ignore[no-any-return]

    def __imod__(self, other: Any) -> H5Array[_T]:
        return self._inplace(np.mod, other)

    def __pow__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return np.array(self) ** other  # type: ignore[no-any-return]

    def __ipow__(self, other: Any) -> H5Array[_T]:
        return self._inplace(np.power, other)

    def __or__(self, other: Any) -> Number | npt.NDArray[Any]:
        return np.array(self) | other  # type: ignore[no-any-return]

    def __ior__(self, other: Any) -> H5Array[_T]:
        return self._inplace(np.logical_or, other)

    def __and__(self, other: Any) -> Number | npt.NDArray[Any]:
        return np.array(self) & other  # type: ignore[no-any-return]

    def __iand__(self, other: Any) -> H5Array[_T]:
        return self._inplace(np.logical_and, other)

    def __invert__(self) -> Number | npt.NDArray[Any]:
        return ~np.array(self)

    def __xor__(self, other: Any) -> Number | npt.NDArray[Any]:
        return np.array(self) ^ other  # type: ignore[no-any-return]

    def __ixor__(self, other: Any) -> H5Array[_T]:
        return self._inplace(np.logical_xor, other)

    # endregion

    # region interface
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        array = np.array(self._dset)

        if dtype is None:
            return array

        return array.astype(dtype)

    def __array_ufunc__(self, ufunc: NP_FUNC, method: str, *inputs: Any, **kwargs: Any) -> Any:
        if method == "__call__":
            if ufunc not in HANDLED_FUNCTIONS:
                return NotImplemented

            return HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs)

        else:
            raise NotImplementedError

    def __array_function__(
        self,
        func: NP_FUNC,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        del types

        if func not in HANDLED_FUNCTIONS:
            return NotImplemented

        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    # endregion

    # region predicates
    @property
    def is_chunked(self) -> bool:
        return self._dset.chunks is not None

    # endregion

    # region attributes
    @property
    def dset(self) -> Dataset[_T] | DatasetWrapper[_T]:
        return self._dset

    @property
    def chunk_size(self) -> int:
        """Get the size of a chunk (i.e. the nb of elements that can be read/written at a time)."""
        return get_size(self.MAX_MEM_USAGE) // self._dset.dtype.itemsize

    @property
    def shape(self) -> tuple[int, ...]:
        return self._dset.shape

    @property
    def dtype(self) -> np.dtype[_T]:
        return self._dset.dtype

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    # endregion

    # region methods
    def _resize(self, amount: int, axis: int | tuple[int, ...] | None = None) -> None:
        if axis is None:
            axis = 0

        if isinstance(axis, int):
            self._dset.resize(self.shape[axis] + amount, axis=axis)

        else:
            self._dset.resize([s + amount if i in axis else s for i, s in enumerate(self.shape)])

    def expand(self, amount: int, axis: int | tuple[int, ...] | None = None) -> None:
        """
        Resize an H5Array by adding `amount` elements along the selected axis.

        Raises:
            TypeError: if the H5Array does not wrap a chunked Dataset.
        """
        self._resize(amount, axis)

    def contract(self, amount: int, axis: int | tuple[int, ...] | None = None) -> None:
        """
        Resize an H5Array by removing `amount` elements along the selected axis.

        Raises:
            TypeError: if the H5Array does not wrap a chunked Dataset.
        """
        self._resize(-amount, axis)

    def astype(self, dtype: npt.DTypeLike, inplace: bool = False) -> H5Array[Any]:
        """
        Cast an H5Array to a specified dtype.
        This does not perform a copy, it returns a wrapper around the underlying H5 dataset.
        """
        if np.issubdtype(dtype, str) and (np.issubdtype(self._dset.dtype, str) or self._dset.dtype == object):
            new_dset = self._dset.asstr()

        else:
            new_dset = self._dset.astype(dtype)

        if inplace:
            file, name = self._dset.file, self._dset.name
            del file[name]

            # FIXME : conversion to np happens anyway but might be expensive, could we save data without conversion ?
            write_dataset(file, name, np.array(new_dset), chunks=new_dset.chunks, maxshape=new_dset.maxshape)

            if file[name].dtype == object:
                self._dset = file[name].asstr()

            else:
                self._dset = file[name]

        return H5Array(new_dset)

    def maptype(self, otype: type[Any]) -> H5Array[Any]:
        """
        Cast an H5Array to any object type.
        This extends H5Array.astype() to any type <T>, where it is required that an object <T> can be constructed as
        T(v) for any value <v> in the dataset.
        """
        return H5Array(self._dset.maptype(otype))

    def iter_chunks(self, keepdims: bool = False) -> ChunkIterator:
        return ChunkIterator(self, keepdims)

    def iter_chunks_with(self, other: npt.NDArray[Any] | H5Array[Any], keepdims: bool = False) -> PairedChunkIterator:
        return PairedChunkIterator(self, other, keepdims)

    def read_direct(
        self,
        dest: npt.NDArray[_T],
        source_sel: tuple[slice, ...],
        dest_sel: tuple[slice, ...],
    ) -> None:
        dset = self._dset.asstr() if np.issubdtype(self.dtype, str) else self._dset
        dset.read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)

    def copy(self) -> npt.NDArray[_T]:
        return np.copy(self)

    def min(self, axis: int | tuple[int, ...] | None = None) -> _T | npt.NDArray[_T]:
        return np.min(self, axis=axis)  # type: ignore[no-any-return]

    def max(self, axis: int | tuple[int, ...] | None = None) -> _T | npt.NDArray[_T]:
        return np.max(self, axis=axis)  # type: ignore[no-any-return]

    def mean(self, axis: int | tuple[int, ...] | None = None) -> Any | npt.NDArray[Any]:
        return np.mean(self, axis=axis)

    # endregion
