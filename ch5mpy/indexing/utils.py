from typing import Callable, Any, Iterable, Generator


def takewhile_inclusive(predicate: Callable[[Any], bool], it: Iterable[Any]) -> Generator[Any, None, None]:
    while True:
        e = next(it, None)  # type: ignore[call-overload]
        yield e

        if e is None or not predicate(e):
            break
