from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable
from ..core.context import RunContext
if TYPE_CHECKING:
    from .pipeline import Pipeline
__all__ = ['Node', 'NodeOutputError']

class NodeOutputError(RuntimeError):
    pass

@dataclass(frozen=True)
class Node:
    name: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    compute: Callable[..., Any]
    tags: tuple[str, ...] = ()

    def forward(self, ctx: RunContext) -> RunContext:
        args = tuple((ctx.require(key) for key in self.inputs))
        result = self.compute(*args)
        if len(self.outputs) == 0:
            return ctx
        if len(self.outputs) == 1:
            return ctx.set(self.outputs[0], result)
        if not isinstance(result, tuple | list):
            raise NodeOutputError(f'Node {self.name!r} declares {len(self.outputs)} outputs but returned non-sequence value {result!r}')
        if len(result) != len(self.outputs):
            raise NodeOutputError(f'Node {self.name!r} declares {len(self.outputs)} outputs {self.outputs!r} but returned {len(result)} values')
        return ctx.set_many(dict(zip(self.outputs, result)))

    def __rshift__(self, other: Any) -> 'Pipeline':
        from .pipeline import Pipeline
        return Pipeline((self,)) >> other
