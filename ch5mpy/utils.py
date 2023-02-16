# coding: utf-8

# ====================================================
# imports
from typing import Any
from typing import Sequence
from typing import Collection
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import TypeGuard


# ====================================================
# code
def is_sequence(obj: Any) -> 'TypeGuard[Sequence[Any]]':
    """Is the object a sequence of objects ? (excluding strings and byte objects.)"""
    return isinstance(obj, Collection) and hasattr(obj, '__getitem__') and not isinstance(
        obj, (str, bytes, bytearray, memoryview)
    )
