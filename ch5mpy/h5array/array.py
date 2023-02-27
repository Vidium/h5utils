# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
import numpy.lib.mixins
from numbers import Number

import numpy.typing as npt
from typing import Any
from typing import Generic
from typing import TypeVar
from typing import Iterator
from typing import TYPE_CHECKING
from ch5mpy._typing import NP_FUNC
from ch5mpy._typing import SELECTOR

from ch5mpy import Dataset
from ch5mpy.h5array.chunks import ChunkIterator
from ch5mpy.h5array.chunks import PairedChunkIterator
from ch5mpy.h5array.indexing.selection import Selection
from ch5mpy.h5array.io import read_one_from_dataset
from ch5mpy.objects.dataset import DatasetWrapper
from ch5mpy.h5array import repr
from ch5mpy.h5array.functions import HANDLED_FUNCTIONS
from ch5mpy.h5array.io import write_to_dataset
from ch5mpy.h5array.indexing.slice import map_slice

if TYPE_CHECKING:
    from ch5mpy.h5array.view import H5ArrayView


# ====================================================
# code
_T = TypeVar("_T", bound=np.generic)


class H5Array(Generic[_T], numpy.lib.mixins.NDArrayOperatorsMixin):
    """Wrapper around Dataset objects to interface with numpy's API."""

    MAX_MEM_USAGE: int | str = "250M"

    # region magic methods
    def __init__(self, dset: Dataset[_T] | DatasetWrapper[_T]):
        if not isinstance(dset, (Dataset, DatasetWrapper)):
            raise TypeError(f"Object of type '{type(dset)}' is not supported by H5Array.")

        if np.issubdtype(np.dtype(str(dset.attrs.get('dtype', 'O'))), str):
            self._dset: Dataset[_T] | DatasetWrapper[_T] = dset.asstr()                       # type: ignore[assignment]

        else:
            self._dset = dset

    def __repr__(self) -> str:
        return (
            f"H5Array({repr.print_dataset(self, end='', padding=8, padding_skip_first=True)}, "
            f"shape={self.shape}, dtype={self._dset.dtype})"
        )

    def __str__(self) -> str:
        return repr.print_dataset(self, sep="")

    def __getitem__(self, index: SELECTOR | tuple[SELECTOR, ...]) -> _T | H5Array[_T] | H5ArrayView[_T]:
        from ch5mpy.h5array.view import H5ArrayView

        selection = Selection.from_selector(index, self.shape)

        if selection.is_empty:
            return H5Array(dset=self._dset)

        elif selection.compute_shape(self.shape) == ():
            return read_one_from_dataset(self._dset, selection, self.dtype)

        else:
            return H5ArrayView(dset=self._dset, sel=selection)

    def __setitem__(self, index: SELECTOR | tuple[SELECTOR, ...], value: Any) -> None:
        selection = Selection.from_selector(index, self.shape)

        try:
            value_arr = np.array(value, dtype=self.dtype)

        except ValueError:
            raise ValueError(f'Could set value of type {type(value)} in H5Array of type {self.dtype}.')

        if np.product(selection.compute_shape(self.shape)) != value_arr.size:
            raise ValueError(f"{' x '.join(map(str, self.shape if selection.is_empty else map(len, selection)))} "
                             f"values were selected but {' x '.join(map(str, value_arr.shape))} were given.")

        if selection.is_empty:
            self._dset[()] = value_arr

        else:
            write_to_dataset(self._dset, value_arr, selection)

    def __len__(self) -> int:
        return len(self._dset)

    def __iter__(self) -> Iterator[_T | npt.NDArray[_T] | H5Array[_T] | H5ArrayView[_T]]:
        for i in range(self.shape[0]):
            yield self[i]

    def __contains__(self, item: Any) -> bool:
        for _, chunk in self.iter_chunks():
            if item in chunk:
                return True

        return False

    def _inplace_operation(self, func: NP_FUNC, value: Any) -> H5Array[_T]:
        if np.issubdtype(self.dtype, str):
            raise TypeError('Cannot perform inplace operation on str H5Array.')

        # special case : 0D array
        if self.shape == ():
            self._dset[:] = func(self._dset[:], value)
            return self

        # general case : 1D+ array
        for index, chunk in self.iter_chunks():
            func(chunk, value, out=chunk)

            # write back result into array
            self._dset.write_direct(chunk, source_sel=map_slice(index, shift_to_zero=True), dest_sel=map_slice(index))

        return self

    def __add__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return self._dset[()] + other                                                      # type: ignore[no-any-return]

    def __iadd__(self, other: Any) -> H5Array[_T]:
        return self._inplace_operation(np.add, other)

    def __sub__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return self._dset[()] - other                                                      # type: ignore[no-any-return]

    def __isub__(self, other: Any) -> H5Array[_T]:
        return self._inplace_operation(np.subtract, other)

    def __mul__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return self._dset[()] * other                                                      # type: ignore[no-any-return]

    def __imul__(self, other: Any) -> H5Array[_T]:
        return self._inplace_operation(np.multiply, other)

    def __truediv__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return self._dset[()] / other                                                      # type: ignore[no-any-return]

    def __itruediv__(self, other: Any) -> H5Array[_T]:
        return self._inplace_operation(np.divide, other)

    def __mod__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return self._dset[()] % other                                                      # type: ignore[no-any-return]

    def __imod__(self, other: Any) -> H5Array[_T]:
        return self._inplace_operation(np.mod, other)

    def __pow__(self, other: Any) -> Number | str | npt.NDArray[Any]:
        return self._dset[()] ** other                                                     # type: ignore[no-any-return]

    def __ipow__(self, other: Any) -> H5Array[_T]:
        return self._inplace_operation(np.power, other)

    def __or__(self, other: Any) -> Number | npt.NDArray[Any]:
        return self._dset[()] | other                                                      # type: ignore[no-any-return]

    def __ior__(self, other: Any) -> H5Array[_T]:
        return self._inplace_operation(np.logical_or, other)

    def __and__(self, other: Any) -> Number | npt.NDArray[Any]:
        return self._dset[()] & other                                                      # type: ignore[no-any-return]

    def __iand__(self, other: Any) -> H5Array[_T]:
        return self._inplace_operation(np.logical_and, other)

    def __invert__(self) -> Number | npt.NDArray[Any]:
        return ~self._dset[()]                                                                  # type: ignore[operator]

    def __xor__(self, other: Any) -> Number | npt.NDArray[Any]:
        return self._dset[()] ^ other                                                      # type: ignore[no-any-return]

    def __ixor__(self, other: Any) -> H5Array[_T]:
        return self._inplace_operation(np.logical_xor, other)

    # endregion

    # region interface
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        array = np.array(self._dset)

        if dtype is None:
            return array

        return array.astype(dtype)

    def __array_ufunc__(self, ufunc: NP_FUNC, method: str, *inputs: Any, **kwargs: Any) \
            -> Any:
        if method == "__call__":
            if ufunc not in HANDLED_FUNCTIONS:
                return NotImplemented

            return HANDLED_FUNCTIONS[ufunc](*inputs, **kwargs)

        else:
            raise NotImplemented

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

    # region attributes
    @property
    def dset(self) -> Dataset[_T] | DatasetWrapper[_T]:
        return self._dset

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
        return self._dset.size

    # endregion

    # region methods
    def astype(self, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
        return np.array(self, dtype=dtype)

    def maptype(self, otype: type[Any]) -> H5Array[Any]:
        return H5Array(self._dset.maptype(otype))

    def iter_chunks(self, keepdims: bool = False) -> ChunkIterator:
        return ChunkIterator(self, keepdims)

    def iter_chunks_with(self,
                         other: npt.NDArray[Any] | H5Array[Any],
                         keepdims: bool = False) -> PairedChunkIterator:
        return PairedChunkIterator(self, other, keepdims)

    def read_direct(self,
                    dest: npt.NDArray[_T],
                    source_sel: tuple[slice, ...],
                    dest_sel: tuple[slice, ...]) -> None:
        dset = self._dset.asstr() if np.issubdtype(self.dtype, str) else self._dset
        dset.read_direct(dest, source_sel=source_sel, dest_sel=dest_sel)

    def copy(self) -> npt.NDArray[_T]:
        return np.copy(self)

    # endregion
