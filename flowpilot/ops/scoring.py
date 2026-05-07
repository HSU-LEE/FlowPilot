from __future__ import annotations
import math
from typing import Callable, Iterable, Sequence, TypeVar
__all__ = ['softmax', 'top_k', 'weighted_sum']
T = TypeVar('T')

def weighted_sum(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights):
        raise ValueError('weighted_sum requires values and weights to have the same length')
    return sum((values[i] * weights[i] for i in range(len(values))))

def softmax(scores: Sequence[float], tau: float=1.0) -> list[float]:
    if tau <= 0:
        raise ValueError('softmax requires tau > 0')
    if not scores:
        return []
    scaled = [s / tau for s in scores]
    m = max(scaled)
    exps = [math.exp(s - m) for s in scaled]
    total = sum(exps)
    return [x / total for x in exps]

def top_k(items: Iterable[T], k: int, key: Callable[[T], float]=lambda x: x) -> list[T]:
    if k < 0:
        raise ValueError('top_k requires k >= 0')
    return sorted(items, key=key, reverse=True)[:k]
