from __future__ import annotations

from typing import Protocol, runtime_checkable, Any
from pathlib import Path

import ch5mpy


@runtime_checkable
class AnonymousArrayCreationFunc(Protocol):
    def __call__(self, name: str, loc: str | Path | ch5mpy.File | ch5mpy.Group) -> ch5mpy.H5Array[Any]:
        ...
