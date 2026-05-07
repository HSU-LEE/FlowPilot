from __future__ import annotations
from typing import TYPE_CHECKING, Any
from .context import RunContext
if TYPE_CHECKING:
    from ..graph.node import Node
    from ..graph.pipeline import Pipeline
__all__ = ['Block']

class Block:
    name: str = ''
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def forward(self, *args: Any) -> Any:
        raise NotImplementedError(f'{type(self).__name__} must implement forward(*inputs)')

    def as_node(self) -> 'Node':
        from ..graph.node import Node
        return Node(name=self.name or type(self).__name__, inputs=self.inputs, outputs=self.outputs, compute=self.forward)

    def __call__(self, ctx: RunContext) -> RunContext:
        return self.as_node().forward(ctx)

    def __rshift__(self, other: Any) -> 'Pipeline':
        return self.as_node() >> other

    def __repr__(self) -> str:
        n = self.name or type(self).__name__
        return f'<Block {n} inputs={self.inputs} outputs={self.outputs}>'
