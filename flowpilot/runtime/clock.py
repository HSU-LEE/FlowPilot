from __future__ import annotations
from dataclasses import dataclass
__all__ = ['Clock']

@dataclass
class Clock:
    dt: float = 1.0
    tick: int = 0
    time: float = 0.0

    def step(self) -> 'Clock':
        self.tick += 1
        self.time += self.dt
        return self

    def reset(self) -> None:
        self.tick = 0
        self.time = 0.0
