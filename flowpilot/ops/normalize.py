from __future__ import annotations
import math
from typing import Sequence
__all__ = ['l2_normalize', 'scale_to_range', 'clamp']

def l2_normalize(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum((x * x for x in v)))
    if norm < 1e-12:
        return list(v)
    return [x / norm for x in v]

def scale_to_range(x: float, lo: float, hi: float, new_lo: float=0.0, new_hi: float=1.0) -> float:
    if hi == lo:
        return new_lo
    t = (x - lo) / (hi - lo)
    return new_lo + t * (new_hi - new_lo)

def clamp(x: float, lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError(f'clamp requires lo <= hi (got lo={lo}, hi={hi})')
    return max(lo, min(hi, x))
