from __future__ import annotations

from types import EllipsisType
from typing import Any, Callable, Iterable, SupportsIndex, Union

from numpy._typing import _ArrayLikeInt_co

SELECTOR = Union[
    None,
    EllipsisType,
    tuple[()],
    int,
    bool,
    SupportsIndex,
    slice,
    range,
    Iterable[int],
    Iterable[bool],
    _ArrayLikeInt_co,
]

NP_FUNC = Callable[..., Any]
H5_FUNC = Callable[..., Any]
