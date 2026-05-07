from __future__ import annotations
from dataclasses import dataclass, field
from ..core.context import RunContext
from ..graph.pipeline import Pipeline
from .cache import TickCache
from .clock import Clock
from .scheduler import Scheduler
__all__ = ['TickLoop']

@dataclass
class TickLoop:
    pipeline: Pipeline
    dt: float = 1.0
    scheduler: Scheduler = field(default_factory=Scheduler)
    cache: TickCache = field(default_factory=TickCache)
    clock: Clock = field(init=False)

    def __post_init__(self) -> None:
        self.clock = Clock(dt=self.dt)

    def run(self, ctx: RunContext | None=None, ticks: int=1, **inputs) -> RunContext:
        if ticks < 0:
            raise ValueError('TickLoop.run requires ticks >= 0')
        if ctx is None:
            ctx = RunContext(inputs)
        elif inputs:
            ctx = ctx.set_many(inputs)
        for _ in range(ticks):
            self.cache.reset()
            ctx = ctx.set_many({'runtime.tick': self.clock.tick, 'runtime.time': self.clock.time, 'runtime.dt': self.clock.dt})
            ctx = self.scheduler.run_due(ctx, self.clock.tick)
            ctx = self.pipeline.run(ctx)
            self.clock.step()
        return ctx
