# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

import numpy as np
import numpy.lib.mixins
from numbers import Number

import numpy.typing as npt
from typing import Any
from typing import cast
from typing import Generic
from typing import TypeVar
from typing import Iterator
from typing import TYPE_CHECKING
from h5utils._typing import SELECTOR
from h5utils._typing import NP_FUNCTION

from h5utils import Dataset
from h5utils.h5array import repr
from h5utils.h5array.inplace import get_chunks
from h5utils.h5array.inplace import get_work_sel
from h5utils.h5array.inplace import get_work_array
from h5utils.h5array.functions import _HANDLED_FUNCTIONS

if TYPE_CHECKING:
    from h5utils.h5array.subset import H5ArrayView

# ====================================================
# code
_T = TypeVar("_T", bound=np.generic)


class H5Array(Generic[_T], numpy.lib.mixins.NDArrayOperatorsMixin):
    """Wrapper around Dataset objects to interface with numpy's API."""

    MAX_MEM_USAGE: int | str = "250M"

    # region magic methods
    def __init__(self, dset: Dataset[_T]):
        if not isinstance(dset, Dataset):
            raise TypeError(
                f"Object of type '{type(dset)}' is not supported by H5Array."
            )

        self._dset = dset

    def __repr__(self) -> str:
        return (
            f"H5Array({repr.print_dataset(self, end='', padding=8, padding_skip_first=True)}, "
            f"shape={self.shape}, dtype={self._dset.dtype})"
        )

    def __str__(self) -> str:
        return repr.print_dataset(self, sep="")

    def __getitem__(self, index: SELECTOR | tuple[SELECTOR]) -> _T | H5Array[_T] | H5ArrayView[_T]:
        from h5utils.h5array.subset import parse_selector, H5ArrayView

        selection, nb_elements = parse_selector(self.shape, index)

        if nb_elements == 1:
            return cast(_T, self._dset[index])

        elif selection is None:
            return H5Array(dset=self._dset)

        else:
            return H5ArrayView(dset=self._dset, sel=selection)

    def __len__(self) -> int:
        return len(self._dset)

    def __iter__(self) -> Iterator[_T | npt.NDArray[_T] | H5Array[_T] | H5ArrayView[_T]]:
        for i in range(self.shape[0]):
            yield self[i]

    def __contains__(self, item: Any) -> bool:
        raise NotImplementedError

    def _inplace_operation(self, func: NP_FUNCTION, value: Any) -> H5Array[_T]:
        if self.shape == ():
            self._dset[:] = func(self._dset[:], value)

        else:
            chunks = get_chunks(H5Array.MAX_MEM_USAGE, self.shape, self.dtype.itemsize)
            work_array = get_work_array(self.shape, chunks[0], dtype=self._dset.dtype)          # type: ignore[type-var]

            for chunk in chunks:
                self._dset.read_direct(
                    work_array, source_sel=chunk, dest_sel=get_work_sel(chunk)
                )
                func(work_array, value, out=work_array)
                self._dset.write_direct(
                    work_array, source_sel=get_work_sel(chunk), dest_sel=chunk
                )

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

    # endregion

    # region interface
    def __array__(self, dtype: npt.DTypeLike | None = None) -> npt.NDArray[Any]:
        return self._dset.astype(dtype)[()]

    def __array_ufunc__(
        self, ufunc: NP_FUNCTION, method: str, *inputs: Any, **kwargs: Any
    ) -> npt.NDArray[_T] | Number | bool:
        if method == "__call__":
            operands: list[Number | Dataset[Any]] = []

            for input in inputs:
                if isinstance(input, Number):
                    operands.append(input)

                elif isinstance(input, H5Array):
                    operands.append(input.dset)

                else:
                    return NotImplemented

            return ufunc(*operands, **kwargs)

        else:
            raise NotImplemented

    def __array_function__(
        self,
        func: NP_FUNCTION,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> npt.NDArray[_T] | Number | bool:
        if func not in _HANDLED_FUNCTIONS:
            return NotImplemented

        return _HANDLED_FUNCTIONS[func](*args, **kwargs)

    # endregion

    # region attributes
    @property
    def dset(self) -> Dataset[_T]:
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

    # endregion

    # region methods
    def astype(self, dtype: npt.DTypeLike) -> npt.NDArray[Any]:
        return np.array(self, dtype=dtype)

    # endregion
