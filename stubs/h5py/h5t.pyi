import numpy as np

from typing import Literal

def string_dtype(
    encoding: Literal["utf-8", "ascii"] = "utf-8", length: int | None = None
) -> np.dtype[np.bytes_] | np.dtype[np.object_]: ...
