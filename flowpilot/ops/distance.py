from __future__ import annotations
import math
from typing import Sequence
__all__ = ['euclidean', 'squared', 'manhattan', 'chebyshev']

def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(sum(((a[i] - b[i]) ** 2 for i in range(len(a)))))

def squared(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(((a[i] - b[i]) ** 2 for i in range(len(a))))

def manhattan(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((abs(a[i] - b[i]) for i in range(len(a))))

def chebyshev(a: Sequence[float], b: Sequence[float]) -> float:
    return max((abs(a[i] - b[i]) for i in range(len(a))))
