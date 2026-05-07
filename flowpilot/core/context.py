from __future__ import annotations
from copy import deepcopy
from typing import Any, Iterator, Mapping
__all__ = ['RunContext']

class RunContext:
    __slots__ = ('_data',)

    def __init__(self, data: Mapping[str, Any] | None=None) -> None:
        self._data: dict[str, Any] = deepcopy(dict(data)) if data else {}

    def get(self, key: str, default: Any=None) -> Any:
        cur: Any = self._data
        for part in key.split('.'):
            if not isinstance(cur, Mapping) or part not in cur:
                return default
            cur = cur[part]
        return cur

    def require(self, key: str) -> Any:
        value = self.get(key, None)
        if value is None and (not self.has(key)):
            raise KeyError(f'RunContext is missing required key: {key!r}')
        return value

    def has(self, key: str) -> bool:
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    def set(self, key: str, value: Any) -> 'RunContext':
        new = RunContext.__new__(RunContext)
        new._data = deepcopy(self._data)
        self._write_path(new._data, key, value)
        return new

    def set_many(self, mapping: Mapping[str, Any]) -> 'RunContext':
        new = RunContext.__new__(RunContext)
        new._data = deepcopy(self._data)
        for key, value in mapping.items():
            self._write_path(new._data, key, value)
        return new

    def update(self, **kwargs: Any) -> 'RunContext':
        return self.set_many(kwargs)

    def keys(self) -> Iterator[str]:
        return iter(self._data.keys())

    def items(self) -> Iterator[tuple[str, Any]]:
        return iter(self._data.items())

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self._data)

    def __contains__(self, key: str) -> bool:
        return self.has(key)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, key: str) -> Any:
        return self.require(key)

    def __repr__(self) -> str:
        return f'RunContext({sorted(self._data.keys())!r})'

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RunContext):
            return NotImplemented
        return self._data == other._data

    @staticmethod
    def _write_path(data: dict[str, Any], key: str, value: Any) -> None:
        parts = key.split('.')
        cur: dict[str, Any] = data
        for part in parts[:-1]:
            nxt = cur.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[part] = nxt
            cur = nxt
        cur[parts[-1]] = value
