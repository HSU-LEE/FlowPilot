from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
from ..core.context import RunContext
__all__ = ['ScheduledTask', 'Scheduler']

@dataclass(frozen=True)
class ScheduledTask:
    name: str
    every: int
    step: Callable[[RunContext], RunContext]

@dataclass
class Scheduler:
    tasks: tuple[ScheduledTask, ...] = ()

    def add(self, name: str, every: int, step: Callable[[RunContext], RunContext]) -> 'Scheduler':
        if every <= 0:
            raise ValueError('Scheduler.add requires every > 0')
        self.tasks = self.tasks + (ScheduledTask(name=name, every=every, step=step),)
        return self

    def run_due(self, ctx: RunContext, tick: int) -> RunContext:
        for task in self.tasks:
            if tick % task.every == 0:
                ctx = task.step(ctx)
        return ctx
