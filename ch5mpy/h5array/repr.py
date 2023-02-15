# coding: utf-8

# ====================================================
# imports
from __future__ import annotations

from typing import Any
from typing import overload
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ch5mpy import H5Array


# ====================================================
# code
@overload
def _get3(arr: H5Array[Any]) -> list[Any | None]:
    ...


@overload
def _get3(arr: None) -> None:
    ...


def _get3(arr: H5Array[Any] | None) -> list[Any | None] | None:
    """Get the first (and last) 3 elements in a set <arr>."""
    if arr is None:
        return None

    if len(arr) <= 6:
        return list(arr[:])

    return list(arr[:3]) + [None] + list(arr[-3:])


def _print3(
    lst: list[Any | None] | None, end: str = "\n", before: str = "", sep: str = ","
) -> str:
    """Print the first (and last) 3 elements in"""
    if lst is None:
        return f"{before}...{end}"

    return f"{before}[{(sep + ' ').join(['...' if e is None else str(e) for e in lst])}]{end}"


def _get_padding(
    padding: int, before: str | None = None, padding_skip_first: bool = False
) -> str:
    """Get the actual needed amount of padding, given that head strings might have been pasted before."""
    if padding_skip_first:
        return ""

    if before is None:
        return " " * padding

    return " " * (padding - len(before))


def _print_dataset_core(
    arr: H5Array[Any] | None,
    padding: int,
    padding_skip_first: bool,
    before: str,
    end: str,
    sep: str,
) -> str:
    # exit condition : array is 1D and can be printed
    if arr is None or arr.ndim == 1:
        return _print3(_get3(arr), before=before, end=end, sep=sep)

    # recursive calls
    rows = _get3(arr)

    spacer = "," + "\n" * (arr.ndim - 1)

    return (
        spacer.join(
            [
                _print_dataset_core(
                    rows[0],
                    padding=padding,
                    padding_skip_first=False,
                    before=_get_padding(padding, before, padding_skip_first)
                    + before
                    + "[",
                    end="",
                    sep=sep,
                )
            ]
            + [
                _print_dataset_core(
                    sub_arr,
                    padding=padding,
                    padding_skip_first=False,
                    before=_get_padding(padding + len(before) + 1),
                    end="",
                    sep=sep,
                )
                for sub_arr in rows[1:-1]
            ]
            + [
                _print_dataset_core(
                    rows[-1],
                    padding=padding,
                    padding_skip_first=False,
                    before=_get_padding(padding + len(before) + 1),
                    end="",
                    sep=sep,
                )
            ]
        )
        + "]"
    )


def print_dataset(
    arr: H5Array[Any],
    padding: int = 0,
    padding_skip_first: bool = False,
    before: str | None = None,
    after: str | None = None,
    end: str = "\n",
    sep: str = ",",
) -> str:
    if arr.ndim == 0:
        array_repr = f"{arr[()]}"

    else:
        array_repr = _print_dataset_core(
            arr,
            padding=padding,
            padding_skip_first=padding_skip_first,
            before="",
            end="",
            sep=sep,
        )

    before = "" if before is None else before
    after = "" if after is None else after

    return f"{before}{array_repr}{after}{end}"
