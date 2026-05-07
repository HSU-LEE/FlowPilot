from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Generic, TypeVar
__all__ = ['Decision']
T = TypeVar('T')

@dataclass(frozen=True)
class Decision(Generic[T]):
    kind: str
    value: T

    def is_(self, kind: str) -> bool:
        return self.kind == kind

    def map(self, fn: Callable[[T], Any]) -> 'Decision[Any]':
        return Decision(kind=self.kind, value=fn(self.value))
