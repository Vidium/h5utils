from contextlib import contextmanager
from typing import Generator, Literal, TypedDict


class _OptionsDict(TypedDict):
    h5dict: Literal["raise", "ignore"]


_ERROR_MODE = _OptionsDict(h5dict="ignore")


def set_error_mode(mode: Literal["raise", "ignore"]) -> None:
    if mode not in ("raise", "ignore"):
        raise ValueError("'mode' must be 'raise' or 'ignore'.")

    _ERROR_MODE["h5dict"] = mode


@contextmanager
def error_mode(mode: Literal["raise", "ignore"]) -> Generator[None, None, None]:
    if mode not in ("raise", "ignore"):
        raise ValueError("'mode' must be 'raise' or 'ignore'.")

    _current_mode = _ERROR_MODE["h5dict"]
    _ERROR_MODE["h5dict"] = mode

    yield

    _ERROR_MODE["h5dict"] = _current_mode
