from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Hashable, TypeVar
__all__ = ['TickCache']
T = TypeVar('T')

@dataclass
class TickCache:
    _store: dict[Hashable, Any] = field(default_factory=dict)

    def get_or_compute(self, key: Hashable, compute: Callable[[], T]) -> T:
        if key in self._store:
            return self._store[key]
        value = compute()
        self._store[key] = value
        return value

    def reset(self) -> None:
        self._store.clear()

    def __contains__(self, key: Hashable) -> bool:
        return key in self._store

    def __getitem__(self, key: Hashable) -> Any:
        return self._store[key]

    def __len__(self) -> int:
        return len(self._store)
