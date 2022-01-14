from typing import Union, Optional, Tuple, List, TypeVar

from .spathlib import SPath


T = TypeVar('T')

Matrix = List[List[T]]

SPathLike = Union[str, SPath]

Range = Union[Optional[int], Tuple[Optional[int], Optional[int]]]
