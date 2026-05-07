from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Iterable
from ..core.context import RunContext
from .node import Node
__all__ = ['Pipeline']

@dataclass(frozen=True)
class Pipeline:
    nodes: tuple[Node, ...]
    name: str = 'pipeline'

    def run(self, ctx: RunContext | None=None, **inputs: Any) -> RunContext:
        if ctx is None:
            ctx = RunContext(inputs)
        elif inputs:
            ctx = ctx.set_many(inputs)
        for node in self.nodes:
            ctx = node.forward(ctx)
        return ctx

    def forward(self, ctx: RunContext) -> RunContext:
        return self.run(ctx)

    def __iter__(self) -> Iterable[Node]:
        return iter(self.nodes)

    def __len__(self) -> int:
        return len(self.nodes)

    def __rshift__(self, other: Any) -> 'Pipeline':
        if isinstance(other, Node):
            return Pipeline(self.nodes + (other,), name=self.name)
        if isinstance(other, Pipeline):
            return Pipeline(self.nodes + other.nodes, name=self.name)
        if hasattr(other, 'as_node'):
            return self >> other.as_node()
        raise TypeError(f'Cannot append {type(other).__name__} to Pipeline')

    def names(self) -> tuple[str, ...]:
        return tuple((node.name for node in self.nodes))
