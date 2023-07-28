from __future__ import annotations

from typing import TYPE_CHECKING, Any, Collection, Sequence

if TYPE_CHECKING:
    from typing_extensions import TypeGuard


def is_sequence(obj: Any) -> TypeGuard[Sequence[Any]]:
    """Is the object a sequence of objects ? (excluding strings and byte objects.)"""
    return (
        isinstance(obj, Collection)
        and hasattr(obj, "__getitem__")
        and not isinstance(obj, (str, bytes, bytearray, memoryview))
    )
