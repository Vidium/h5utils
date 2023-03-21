# coding: utf-8
# Created on 16/01/2023 11:09
# Author : matteo

# ====================================================
# imports
from __future__ import annotations

from enum import Enum


# ====================================================
# code
class H5Mode(str, Enum):
    READ = "r"
    READ_WRITE = "r+"
    WRITE_TRUNCATE = "w"
    WRITE = "w-"
    READ_WRITE_CREATE = "a"
#
#     @classmethod
#     def get(cls, mode: str) -> H5Mode:
#         return _REV[mode]
#
# _REV = {getattr(H5Mode, v): v for v in vars(H5Mode) if not v.startswith('__')}
