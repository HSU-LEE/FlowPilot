from __future__ import annotations
import math
from typing import Sequence
__all__ = ['wrap_to_pi', 'angle_between', 'bearing_to', 'bearing_within']

def wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi

def angle_between(a: float, b: float) -> float:
    return wrap_to_pi(b - a)

def bearing_to(origin: Sequence[float], target: Sequence[float]) -> float:
    return math.atan2(target[1] - origin[1], target[0] - origin[0])

def bearing_within(origin: Sequence[float], heading: float, target: Sequence[float], tol: float) -> bool:
    return abs(angle_between(heading, bearing_to(origin, target))) < tol
